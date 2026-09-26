"""Runs the FeatureSelector across all rules; returns detected hits only."""

import pandas as pd
from .library import ReferenceLibrary
from .selector import FeatureSelector


class FunctionalGroupClassifier:
    def __init__(self, library: ReferenceLibrary, selector: FeatureSelector):
        self.library  = library
        self.selector = selector

    def classify(self, spectrum: pd.Series) -> list:
        return [wr for rule in self.library.rules
                for wr in [self.selector.extract(spectrum, rule)]
                if wr.detected]
