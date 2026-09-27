"""
Spectrum pre-processing: Savitzky-Golay smoothing + Asymmetric Least Squares
(ALS) baseline correction. Runs BEFORE any detection strategy (rule-based
FeatureSelector or DeconvolutionSelector) ever sees the spectrum.

Design decision (carried over from the DeconvolutionSelector discussion):
DeconvolutionSelector's Gaussian fit keeps its own linear baseline term
(m*x + b) regardless of whether ALS runs here. ALS is a toggle the user can
turn off in the sidebar; the linear term is cheap insurance for whatever
residual slope ALS leaves behind (or the *only* baseline handling, if ALS
is off). The two are not meant to be mutually exclusive.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve


class SpectrumPreprocessor:
    """
    process(spectrum) -> cleaned spectrum, same index, same length.

    smooth / baseline_correct are independent toggles — either, both, or
    neither can run, so a Streamlit sidebar can expose them as two checkboxes
    without the class caring which combination was chosen.
    """

    def __init__(self,
                 smooth: bool = True,
                 sg_window: int = 15,
                 sg_polyorder: int = 3,
                 baseline_correct: bool = False,
                 als_lam: float = 1e5,
                 als_p: float = 0.01,
                 als_niter: int = 10):
        if sg_window % 2 == 0:
            raise ValueError("sg_window must be odd (Savitzky-Golay requires an odd window).")
        if sg_polyorder >= sg_window:
            raise ValueError("sg_polyorder must be smaller than sg_window.")

        self.smooth = smooth
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder

        self.baseline_correct = baseline_correct
        self.als_lam = als_lam
        self.als_p = als_p
        self.als_niter = als_niter

    def process(self, spectrum: pd.Series) -> pd.Series:
        y = spectrum.values.astype(float)

        if self.smooth:
            y = self._smooth(y)

        if self.baseline_correct:
            baseline = self._als_baseline(y, lam=self.als_lam, p=self.als_p, niter=self.als_niter)
            y = y - baseline

        return pd.Series(y, index=spectrum.index)

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        window = self.sg_window
        # Guard against spectra shorter than the configured window (edge case,
        # not expected for real FTIR data, but cheap to protect against).
        if window >= len(y):
            window = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
            if window <= self.sg_polyorder:
                return y  # too short to smooth meaningfully; return unchanged
        return savgol_filter(y, window_length=window, polyorder=self.sg_polyorder)

    @staticmethod
    def _als_baseline(y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
        """
        Asymmetric Least Squares baseline (Eilers & Boelens, 2005).
        lam  : smoothness penalty — higher = smoother/stiffer baseline
        p    : asymmetry — lower = baseline hugs the bottom of the spectrum
               (appropriate for absorbance data, where peaks point up)
        niter: reweighting iterations
        """
        L = len(y)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
        D = lam * D.dot(D.T)
        w = np.ones(L)
        z = y.copy()
        for _ in range(niter):
            W = diags(w, 0, shape=(L, L), dtype=float)
            Z = csc_matrix(W + D)
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
