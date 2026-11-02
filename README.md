# OpenFTIR
An attempt to leverage Data Analytics and Polymer Science intersection to create a platform for faster FTIR spectrum analysis, thus making a Context-Aware Spectrum Analyzer and Elevating materials informatics through open-source, publication-quality spectral analysis.

### Overview 

OpenFTIR takes raw ATR-FTIR absorbance spectra and a reference peak-assignment library, and returns a structured, per-sample functional-group profile — no manual peak-picking required. It's built for polymer scientists who need to screen many spectra against known band assignments (C=O stretching, N–H bending, aromatic ring modes, etc.) without re-deriving thresholds by eye each time.

The current scaffold (`ftir_analyzer.py`) implements the core pipeline end-to-end: reference parsing → windowed peak extraction → intensity-tiered detection → per-sample and cross-sample reporting.

### Why

Manual FTIR interpretation doesn't scale. A researcher comparing dozens of polymer blends, additive formulations, or degradation time-points against a reference library ends up re-checking the same absorbance windows over and over, with detection thresholds that live in someone's head rather than in code. OpenFTIR externalizes that knowledge into a versioned, auditable reference library and a reproducible detection pipeline, so the same calibration is applied consistently across every sample and every run.

### How it works

```
Reference CSV ──▶ ReferenceLibrary        ---(parses peak windows + qualitative intensity)
                        │
Spectra CSV   ──▶ CompoundProfiler.load_spectra()
                        │
                        ▼
              FunctionalGroupClassifier
                        │  (runs FeatureSelector per rule, per sample)
                        ▼
                  WindowResult(s) ──▶ CompoundProfile ──▶ summary() / presence_matrix()
```

1. **`ReferenceLibrary`** parses a reference CSV of known peak assignments. Peak positions may be given as a range (`"1705-1725"`) or a single wavenumber (`"1760"`, auto-expanded to ±5 cm⁻¹). Rows with unparseable positions are skipped rather than raising.
2. **`FeatureSelector`** slices each spectrum to the window defined by a `PeakRule`, and reports the peak and mean absorbance within it. Detection is decided against a calibrated absorbance threshold keyed to the rule's qualitative intensity tier (`very_strong`, `strong`, `medium`, `weak`).
3. **`FunctionalGroupClassifier`** runs the selector across every rule in the library for a given spectrum and keeps only the hits.
4. **`CompoundProfiler`** is the top-level orchestrator: it loads a sample × wavenumber spectra table, profiles one or all samples, and can build a binary sample × compound-class presence matrix for cross-sample comparison.

### Intensity calibration

Detection thresholds are calibrated from ATR-FTIR polymer data and expressed in absorbance units:

| Tier | Peak absorbance |
|---|---|
| `very_strong` | > 0.30 A |
| `strong` | > 0.05 A |
| `medium` | > 0.015 A |
| `weak` | > 0.003 A |

These are read off each `PeakRule`'s qualitative `peak_details` field (e.g. "strong, sharp") and mapped to a numeric floor, so a reference library authored in plain chemist-readable language drives quantitative detection without extra annotation work.

### Installation

```bash
git clone https://github.com/<your-org>/OpenFTIR.git
cd OpenFTIR
pip install pandas numpy
```

No other dependencies are required for Version 0 Scaffold.

### Input file formats

**Reference library CSV** — row per known peak assignment:

| Column | Description |
|---|---|
| `Peak Position` | Wavenumber range (`"1705-1725"`) or single value (`"1760"`) |
| `Group` | Vibration type, e.g. `"C=O stretching"` |
| `Class` | Chemical class, e.g. `"aliphatic ketone"` |
| `Peak Details` | Qualitative intensity/shape, e.g. `"strong, sharp"` |

**Spectra CSV** — one row per sample, one column per wavenumber:

| `Sample_Name` | 4000 | 3998 | ... | 400 |
|---|---|---|---|---|
| Nylon (6,6) GF | 0.012 | 0.013 | ... | 0.041 |

## Quick start

```python
from ftir_analyzer import ReferenceLibrary, CompoundProfiler

lib = ReferenceLibrary("reference_data.csv")
profiler = CompoundProfiler(lib)
profiler.load_spectra("sample_spectra.csv")

# Single sample
profile = profiler.profile_sample("Nylon (6,6) GF")
print(profile.summary())

# All samples + cross-sample presence matrix
all_profiles = profiler.profile_all()
matrix = profiler.presence_matrix(all_profiles)
```

Running the module directly (`python ftir_analyzer.py`) executes a demo pass: it loads the bundled reference/spectra CSVs, prints a hit-count summary per sample, a detailed report for a representative sample, and a presence/absence matrix for compound classes detected in ≥5 samples.

## Output

`CompoundProfile.summary()` returns a `pandas.DataFrame`, one row per detected peak, sorted by peak absorbance:

| range_cm⁻¹ | peak_cm⁻¹ | group | class | intensity_tier | peak_absorbance |
|---|---|---|---|---|---|
| 1705–1725 | 1718 | C=O stretching | aliphatic ketone | strong | 0.0842 |

`CompoundProfiler.presence_matrix()` returns a binary sample × compound-class `DataFrame`, useful for clustering, PCA, or cross-formulation comparison.

## Roadmap

- [ ] Baseline correction and spectral smoothing pre-processing
- [ ] Peak deconvolution for overlapping bands
- [ ] Configurable/overridable intensity thresholds per compound class
- [ ] Automated reference-library validation and conflict detection
- [ ] Interactive spectrum viewer with detected-peak overlays
- [ ] Export to publication-ready figures (Matplotlib/Plotly)
- [ ] Batch reporting (PDF/HTML) across a full sample set

## Contributing

Issues and PRs are welcome — particularly reference-library contributions (peer-reviewed peak assignments) and threshold calibration data from additional instrument/polymer combinations. Please cite the source of any new peak assignments in the PR description.

## License

TBD.
