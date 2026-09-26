"""OpenFTIR — context-aware FTIR spectrum analyzer."""

__version__ = "1.0.0"

from .models import PeakRule, WindowResult, CompoundProfile
from .library import ReferenceLibrary
from .selector import FeatureSelector
from .classifier import FunctionalGroupClassifier
from .profiler import CompoundProfiler
from .plotting import standardize_to_absorbance, plot_single, plot_compare, load_data

__all__ = [
    "PeakRule", "WindowResult", "CompoundProfile",
    "ReferenceLibrary", "FeatureSelector", "FunctionalGroupClassifier", "CompoundProfiler",
    "standardize_to_absorbance", "plot_single", "plot_compare", "load_data",
]
