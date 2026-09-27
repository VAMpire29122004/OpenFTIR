"""Runs a Selector across all rules; returns detected hits only."""

from typing import Protocol
import pandas as pd
from .library import ReferenceLibrary
from .models import PeakRule, WindowResult


class Selector(Protocol):
    """Structural type: anything with this method works — FeatureSelector,
    DeconvolutionSelector, or any future selector, without inheritance."""
    def extract(self, spectrum: pd.Series, rule: PeakRule) -> WindowResult: ...


class FunctionalGroupClassifier:
    def __init__(self, library: ReferenceLibrary, selector: Selector):
        self.library  = library
        self.selector = selector

    def classify(self, spectrum: pd.Series) -> list:
        return [wr for rule in self.library.rules
                for wr in [self.selector.extract(spectrum, rule)]
                if wr.detected]
