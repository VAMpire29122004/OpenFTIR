"""Top-level orchestrator: loads spectra, drives classification, builds profiles."""

from typing import Optional
import pandas as pd
from .library import ReferenceLibrary
from .selector import FeatureSelector
from .classifier import FunctionalGroupClassifier, Selector
from .preprocessing import SpectrumPreprocessor
from .models import CompoundProfile


class CompoundProfiler:
    def __init__(self, library: ReferenceLibrary, selector: Optional[Selector] = None,
                 preprocessor: Optional[SpectrumPreprocessor] = None):
        self.library      = library
        self.selector     = selector or FeatureSelector()
        self.classifier   = FunctionalGroupClassifier(library, self.selector)
        self.preprocessor = preprocessor  # None = no preprocessing, spectra used as-is
        self.spectra_df: Optional[pd.DataFrame] = None

    def load_spectra(self, path_or_buffer, transpose: bool = False):
        if transpose:
            df = pd.read_csv(path_or_buffer, index_col=0).T
        else:
            df = pd.read_csv(path_or_buffer).set_index("Sample_Name")
        df.columns = df.columns.astype(float)
        self.spectra_df = df

    def get_spectrum(self, sample_name: str) -> pd.Series:
        """Returns the spectrum for one sample, preprocessed if a preprocessor
        is set. Separated out from profile_sample so plotting (next checkpoint)
        can request the exact same cleaned spectrum that detection ran on,
        rather than re-deriving it separately."""
        if self.spectra_df is None:
            raise RuntimeError("Call load_spectra() first.")
        spectrum = self.spectra_df.loc[sample_name]
        spectrum.index = spectrum.index.astype(float)
        if self.preprocessor is not None:
            spectrum = self.preprocessor.process(spectrum)
        return spectrum

    def profile_sample(self, sample_name: str) -> CompoundProfile:
        spectrum = self.get_spectrum(sample_name)
        detections = self.classifier.classify(spectrum)
        return CompoundProfile(sample_name, detections)

    def profile_all(self) -> dict:
        return {name: self.profile_sample(name) for name in self.spectra_df.index}

    def presence_matrix(self, profiles: dict) -> pd.DataFrame:
        """Binary sample x detected-class matrix."""
        all_classes = sorted({wr.rule.compound_class
                               for p in profiles.values()
                               for wr in p.detections})
        mat = pd.DataFrame(0, index=profiles.keys(), columns=all_classes)
        for name, prof in profiles.items():
            for wr in prof.detections:
                mat.loc[name, wr.rule.compound_class] = 1
        return mat


    def presence_matrix(self, profiles: dict) -> pd.DataFrame:
        """Binary sample x detected-class matrix."""
        all_classes = sorted({wr.rule.compound_class
                               for p in profiles.values()
                               for wr in p.detections})
        mat = pd.DataFrame(0, index=profiles.keys(), columns=all_classes)
        for name, prof in profiles.items():
            for wr in prof.detections:
                mat.loc[name, wr.rule.compound_class] = 1
        return mat
