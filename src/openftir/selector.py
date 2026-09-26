"""
Feature selectors: given one spectrum and one PeakRule, decide whether the
peak is present and where. v1 ships the threshold-based FeatureSelector only.

DeconvolutionSelector (Gaussian curve fitting for overlapping bands) is
[Unreleased] — see CHANGELOG — and will land in this module for v2 once
SpectrumPreprocessor is in place. Both selectors implement the same
`extract(spectrum, rule) -> WindowResult` interface, so FunctionalGroupClassifier
never needs to know which one it's driving.
"""

import pandas as pd
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
