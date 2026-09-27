"""
Feature selectors: given one spectrum and one PeakRule, decide whether the
peak is present and where. Two implementations, same interface:

  - FeatureSelector: simple max-in-window thresholding (v1)
  - DeconvolutionSelector: Gaussian curve fitting to separate overlapping
    bands, with a linear baseline term fit jointly (v2)

Both implement `extract(spectrum, rule) -> WindowResult`, so
FunctionalGroupClassifier / CompoundProfiler never need to know which one
they're driving.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from .models import PeakRule, WindowResult


class FeatureSelector:
    """
    Extracts peak statistics for one PeakRule from one spectrum.
    Spectra are absorbance values; thresholds are in absorbance units.
    """

    # Calibrated from polymer ATR-FTIR data distribution
    THRESHOLDS = {
        "very_strong" : 0.30,
        "strong"      : 0.05,
        "medium"      : 0.015,
        "weak"        : 0.003,
    }

    def extract(self, spectrum: pd.Series, rule: PeakRule) -> WindowResult:
        """
        spectrum : pd.Series  index=wavenumber(float), values=absorbance
        """
        mask   = (spectrum.index >= rule.wn_low) & (spectrum.index <= rule.wn_high)
        window = spectrum[mask]

        if window.empty:
            return WindowResult(rule, 0.0, 0.0, rule.wn_low, False)

        peak_abs  = float(window.max())
        mean_abs  = float(window.mean())
        peak_wn   = float(window.idxmax())
        threshold = self.THRESHOLDS.get(rule.intensity_tier, 0.015)
        detected  = peak_abs >= threshold

        return WindowResult(rule, peak_abs, mean_abs, peak_wn, detected)


def _multi_gaussian_baseline(wavenumbers, m, b, *params):
    """Linear baseline (m*x + b) plus a sum of Gaussian curves.
    params grouped in threes: (amplitude, center, width) per peak."""
    y = m * wavenumbers + b
    for i in range(0, len(params), 3):
        amp, ctr, wid = params[i:i + 3]
        y += amp * np.exp(-((wavenumbers - ctr) / wid) ** 2)
    return y


class DeconvolutionSelector:
    """
    Extracts peak statistics using topological prominence and Gaussian curve
    fitting with a linear baseline, to separate overlapping functional groups.
    """

    THRESHOLDS = FeatureSelector.THRESHOLDS

    def __init__(self, prominence: float = 0.005, width_cm: float = 2.0, pad: float = 20.0):
        self.prominence = prominence
        self.width_cm   = width_cm
        self.pad        = pad
        # TODO (v2, later checkpoint): memoization cache to avoid refitting
        # the same overlapping region once for every rule that lands in it.

    def extract(self, spectrum: pd.Series, rule: PeakRule) -> WindowResult:
        spec_sorted = spectrum.sort_index()
        x = spec_sorted.index.values
        y = spec_sorted.values

        dx = np.median(np.abs(np.diff(x)))
        if dx == 0:
            dx = 1.0
        width_samples = max(1, self.width_cm / dx)

        peaks, properties = find_peaks(y, prominence=self.prominence, width=width_samples)
        valid_peaks = [p for p in peaks if rule.wn_low <= x[p] <= rule.wn_high]

        threshold = self.THRESHOLDS.get(rule.intensity_tier, 0.015)

        if not valid_peaks:
            mask = (x >= rule.wn_low) & (x <= rule.wn_high)
            mean_abs = float(y[mask].mean()) if mask.any() else 0.0
            return WindowResult(rule, 0.0, mean_abs, rule.wn_low, False)

        region_mask = (x >= (rule.wn_low - self.pad)) & (x <= (rule.wn_high + self.pad))
        x_region = x[region_mask]
        y_region = y[region_mask]

        if len(x_region) < 3:
            # Fixed from the earlier draft: this used to hardcode detected=True,
            # which meant any rule whose window fell near a spectrum edge (too
            # short a region to fit) got an automatic positive regardless of
            # actual peak absorbance. Now applies the same threshold as every
            # other path.
            target_peak_idx = valid_peaks[np.argmax(y[valid_peaks])]
            peak_abs = float(y[target_peak_idx])
            return WindowResult(rule, peak_abs, float(y_region.mean()),
                                 float(x[target_peak_idx]), peak_abs >= threshold)

        m_guess = ((y_region[-1] - y_region[0]) / (x_region[-1] - x_region[0])
                   if (x_region[-1] - x_region[0]) != 0 else 0)
        b_guess = y_region[0] - m_guess * x_region[0]

        guess        = [m_guess, b_guess]
        bounds_lower = [-np.inf, -np.inf]
        bounds_upper = [np.inf, np.inf]

        peak_widths_cm = properties["widths"] * dx
        for p, w_cm in zip(peaks, peak_widths_cm):
            if (rule.wn_low - self.pad) <= x[p] <= (rule.wn_high + self.pad):
                amp_guess, ctr_guess = y[p], x[p]
                guess.extend([amp_guess, ctr_guess, w_cm])
                bounds_lower.extend([0, ctr_guess - 15, 0.1])
                bounds_upper.extend([np.inf, ctr_guess + 15, 200])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _multi_gaussian_baseline, x_region, y_region,
                    p0=guess, bounds=(bounds_lower, bounds_upper), maxfev=2500,
                )

            best_amp, best_ctr = 0.0, rule.wn_low
            for i in range(2, len(popt), 3):
                amp, ctr, _ = popt[i:i + 3]
                if rule.wn_low <= ctr <= rule.wn_high and amp > best_amp:
                    best_amp, best_ctr = amp, ctr

            peak_abs, peak_wn = best_amp, best_ctr

        except (RuntimeError, ValueError):
            target_peak_idx = valid_peaks[np.argmax(y[valid_peaks])]
            peak_abs = float(y[target_peak_idx])
            peak_wn  = float(x[target_peak_idx])

        mean_abs = float(y[(x >= rule.wn_low) & (x <= rule.wn_high)].mean())
        detected = peak_abs >= threshold

        return WindowResult(rule, peak_abs, mean_abs, peak_wn, detected)

