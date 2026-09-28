"""
Core data models shared by every detection strategy (rule-based, deconvolution, ...).
Nothing here should ever need to change when a new FeatureSelector variant is added.
"""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PeakRule:
    wn_low: float
    wn_high: float
    group: str           # vibration type  e.g. "C=O stretching"
    compound_class: str  # chemical class  e.g. "aliphatic ketone"
    peak_details: str    # qualitative     e.g. "strong, sharp"

    @property
    def intensity_tier(self) -> str:
        # Normalized so this works whether peak_details is natural-language
        # CSV text ("strong, sharp") OR an already-computed tier keyword
        # ("very_strong") -- the latter is needed when reconstructing a
        # PeakRule from edited/resolved peak data (see interactive.py).
        # Without the underscore->space normalization, "very_strong" would
        # fail the "very strong" check and silently fall through to match
        # "strong" instead (it's a substring of "very_strong").
        d = self.peak_details.lower().replace("_", " ")
        if "manual" in d:
            return "manual"
        if "very strong" in d:
            return "very_strong"
        if "strong" in d:
            return "strong"
        if "medium" in d:
            return "medium"
        return "weak"


@dataclass
class WindowResult:
    rule: PeakRule
    peak_absorbance: float    # max absorbance in the window
    mean_absorbance: float
    peak_wavenumber: float    # wavenumber of peak maximum
    detected: bool


@dataclass
class CompoundProfile:
    sample_name: str
    detections: list = field(default_factory=list)  # list[WindowResult]

    def summary(self) -> pd.DataFrame:
        rows = []
        for w in self.detections:
            rows.append({
                "range_cm⁻¹"      : f"{w.rule.wn_low:.0f}–{w.rule.wn_high:.0f}",
                "peak_cm⁻¹"       : round(w.peak_wavenumber, 0),
                "group"           : w.rule.group,
                "class"           : w.rule.compound_class,
                "intensity_tier"  : w.rule.intensity_tier,
                "peak_absorbance" : round(w.peak_absorbance, 4),
            })
        return pd.DataFrame(rows).sort_values("peak_absorbance", ascending=False)
