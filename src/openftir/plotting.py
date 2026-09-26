"""
FTIR Publication-Quality Spectrum Plotter
==========================================
Rendering only. All peak detection is delegated to ReferenceLibrary +
FeatureSelector (via FunctionalGroupClassifier) — this module no longer
parses the reference CSV or applies intensity thresholds itself, so there
is exactly one place detection logic can be wrong, not three.

Merges what used to be two divergent scripts:
  - collision-aware label placement + leader-line annotations + shaded
    intensity-tier regions (from the earlier "no overlapping labels" version)
  - auto absorbance/transmittance conversion + configurable y_out display
    format (from the version wired into the Streamlit app)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches as mpatches
from matplotlib import rcParams

from .library import ReferenceLibrary
from .selector import FeatureSelector
from .classifier import FunctionalGroupClassifier
from .models import WindowResult

# ── Publication font settings ──────────────────────────────────────────────
rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size"          : 9,
    "axes.labelsize"     : 11,
    "axes.titlesize"     : 12,
    "xtick.labelsize"    : 9,
    "ytick.labelsize"    : 9,
    "legend.fontsize"    : 8,
    "figure.dpi"         : 300,
    "savefig.dpi"        : 300,
    "savefig.bbox"       : "tight",
    "savefig.facecolor"  : "white",
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.linewidth"     : 0.8,
    "xtick.direction"    : "in",
    "ytick.direction"    : "in",
    "xtick.major.size"   : 4,
    "ytick.major.size"   : 4,
    "xtick.minor.size"   : 2,
    "ytick.minor.size"   : 2,
    "xtick.minor.visible": True,
})

REGION_COLORS = {
    "very_strong": "#d62728",
    "strong"     : "#ff7f0e",
    "medium"     : "#2ca02c",
    "weak"       : "#9467bd",
}

COMPARE_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


# ── Data loading & y-axis conversion ────────────────────────────────────────

def load_data(spectra_path, ref_path, transpose=False):
    """CLI/notebook convenience loader. Streamlit app builds df/library directly."""
    if transpose:
        df = pd.read_csv(spectra_path, index_col=0).T
    else:
        df = pd.read_csv(spectra_path).set_index("Sample_Name")
    df.columns = df.columns.astype(float)
    library = ReferenceLibrary(ref_path)
    return df, library


def standardize_to_absorbance(df: pd.DataFrame, mode: str = "auto") -> pd.DataFrame:
    """Ensures all internal data is Absorbance for accurate peak detection."""
    if mode == "auto":
        max_val = df.values.max()
        min_val = df.values.min()
        if max_val <= 1.2 and min_val >= -0.1:
            mode = "trans_frac"
        elif max_val > 5.0:
            mode = "trans_pct"
        else:
            mode = "abs"

    if mode == "trans_frac":
        df_safe = np.clip(df, a_min=1e-7, a_max=None)
        return -np.log10(df_safe)
    elif mode == "trans_pct":
        df_safe = np.clip(df / 100.0, a_min=1e-7, a_max=None)
        return -np.log10(df_safe)
    return df


def _to_display(ab, y_out: str):
    """Converts an absorbance value/array to the requested display scale.
    NOTE: this was missing in the earlier collision-aware version, which
    only ever drew in absorbance — without it, leader-line arrows would
    point at the wrong y-position whenever y_out != 'abs'."""
    if y_out == "trans_pct":
        return 100.0 * (10.0 ** -ab)
    if y_out == "trans_frac":
        return 10.0 ** -ab
    return ab


def _ylabel_for(y_out: str, offset: bool = False) -> str:
    suffix = " (offset)" if offset else ""
    if y_out == "trans_pct":
        return f"Transmittance{suffix} (%)"
    if y_out == "trans_frac":
        return f"Transmittance{suffix} (fraction)"
    return f"Absorbance{suffix} (a.u.)"


# ── Detection: delegates entirely to ReferenceLibrary / FeatureSelector ────

def _detect(spectrum: pd.Series, library: ReferenceLibrary,
            selector: FeatureSelector, min_dist: float = 30.0) -> list:
    """Runs classification, then suppresses near-duplicate hits (multiple
    rules covering the same physical peak) — the plotting-side counterpart
    to FeatureSelector's per-rule thresholding."""
    classifier = FunctionalGroupClassifier(library, selector)
    hits = classifier.classify(spectrum)
    hits.sort(key=lambda w: w.peak_absorbance, reverse=True)

    kept = []
    for h in hits:
        too_close = any(abs(h.peak_wavenumber - k.peak_wavenumber) < min_dist for k in kept)
        if not too_close:
            kept.append(h)
    return kept


def _short_label(hit: WindowResult) -> str:
    """Two-line label: class on top, abbreviated vibration mode below."""
    abbr = (hit.rule.group
            .replace("stretching", "str.")
            .replace("bending", "bend.")
            .replace("out-of-plane", "oop")
            .replace("(aromatic)", "(ar.)")
            .replace("(hydrogen-bonded)", "(H-bond)")
            .replace("(free)", "(free)"))
    return f"{hit.rule.compound_class}\n{abbr}"


# ── Collision-aware label placement ─────────────────────────────────────────

def _resolve_label_positions(hits: list, y_base: float, y_step: float,
                              min_wn_gap: float = 120.0, max_iter: int = 80) -> list:
    """
    Assigns (x, y) text positions for each hit so labels don't overlap.
      1. Start each label at its peak wavenumber, alternating two y-rows.
      2. Iteratively push overlapping labels apart along x.
      3. Clamp x to the plot's wavenumber range.
    Returns list of dicts: {x, y, hit}.
    """
    if not hits:
        return []

    ordered = sorted(hits, key=lambda h: h.peak_wavenumber, reverse=True)

    positions = []
    for i, h in enumerate(ordered):
        row   = i % 2
        y_pos = y_base + (1 - row) * y_step
        positions.append({"x": h.peak_wavenumber, "y": y_pos, "hit": h})

    for _ in range(max_iter):
        moved = False
        positions.sort(key=lambda p: p["x"], reverse=True)
        for i in range(len(positions) - 1):
            pi, pj = positions[i], positions[i + 1]
            gap = pi["x"] - pj["x"]
            if gap < min_wn_gap:
                push = (min_wn_gap - gap) / 2.0
                pi["x"] += push
                pj["x"] -= push
                moved = True
        if not moved:
            break

    for p in positions:
        p["x"] = max(600.0, min(4050.0, p["x"]))

    return positions


# ── Single-sample publication plot ──────────────────────────────────────────

def plot_single(sample_name: str, df: pd.DataFrame, library: ReferenceLibrary,
                 selector: FeatureSelector = None, out_path: str = None,
                 max_labels: int = 12, title: str = None, y_out: str = "abs"):
    selector = selector or FeatureSelector()

    spectrum = df.loc[sample_name].sort_index(ascending=False)
    spectrum.index = spectrum.index.astype(float)
    wn = spectrum.index.values.astype(float)
    ab = spectrum.values.astype(float)  # always absorbance internally

    hits = _detect(spectrum, library, selector)
    top  = hits[:max_labels]

    display_y = _to_display(ab, y_out)
    ylabel    = _ylabel_for(y_out)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(wn, display_y, color="#1a1a2e", linewidth=0.9, zorder=5)

    ax_ymax = float(np.max(display_y)) * 1.05
    ax_ymin = float(max(np.min(display_y) - np.max(display_y) * 0.03, 0))

    # Shaded intensity-tier regions
    seen = set()
    for h in top:
        key = (round(h.rule.wn_low), round(h.rule.wn_high))
        if key in seen:
            continue
        seen.add(key)
        c = REGION_COLORS[h.rule.intensity_tier]
        ax.axvspan(h.rule.wn_low, h.rule.wn_high, alpha=0.10, color=c, zorder=1)
        ax.axvline(h.peak_wavenumber, color=c, linewidth=0.5,
                   alpha=0.55, linestyle="--", zorder=2)

    label_area_frac = 0.55
    ax.set_xlim(4100, 580)
    total_y = ax_ymax / (1.0 - label_area_frac)
    ax.set_ylim(ax_ymin, ax_ymax + total_y * label_area_frac)

    y_base = ax_ymax * 1.08
    y_step = ax_ymax * 0.22
    label_positions = _resolve_label_positions(top, y_base=y_base, y_step=y_step, min_wn_gap=130.0)

    for lp in label_positions:
        h = lp["hit"]
        color = REGION_COLORS[h.rule.intensity_tier]
        peak_display_y = _to_display(h.peak_absorbance, y_out)

        ax.annotate(
            _short_label(h),
            xy=(h.peak_wavenumber, peak_display_y),
            xytext=(lp["x"], lp["y"]),
            fontsize=6.5, ha="center", va="bottom", color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.75,
                             alpha=0.85, connectionstyle="arc3,rad=0.0"),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=0.6, alpha=0.97),
            zorder=10,
        )

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(4000, 500, -500))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100))

    ax.set_title(title or f"ATR-FTIR Spectrum — {sample_name}", fontsize=12, pad=10, fontweight="bold")

    legend_patches = [
        mpatches.Patch(facecolor=REGION_COLORS[t], alpha=0.7, label=t.replace("_", " ").title())
        for t in ["very_strong", "strong", "medium", "weak"]
        if any(h.rule.intensity_tier == t for h in top)
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, title="Peak intensity",
                   title_fontsize=7.5, fontsize=7, loc="upper left",
                   framealpha=0.9, edgecolor="#cccccc")

    ax.axvline(1500, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=0)
    for label, xpos in [("Functional group region", 2800), ("Fingerprint region", 1050)]:
        ax.text(xpos, ax_ymin - ax_ymax * 0.03, label, ha="center", va="top",
                 fontsize=7.5, color="#999999", style="italic")

    plt.tight_layout()
    out = out_path or f"{sample_name.replace(' ', '_').replace('/', '_')}_FTIR.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ── Multi-sample comparison (stacked offset) ────────────────────────────────

def plot_compare(sample_names: list, df: pd.DataFrame, library: ReferenceLibrary,
                  selector: FeatureSelector = None, out_path: str = None,
                  title: str = None, offset: float = None,
                  annotate_first: bool = True, y_out: str = "abs"):
    selector = selector or FeatureSelector()
    n = len(sample_names)
    fig, ax = plt.subplots(figsize=(11, 3.5 + 1.5 * n))

    spectra = []
    for name in sample_names:
        sp = df.loc[name].sort_index(ascending=False)
        sp.index = sp.index.astype(float)
        spectra.append(sp)

    step = 0
    ylabel = _ylabel_for(y_out, offset=True)

    for i, (name, sp) in enumerate(zip(sample_names, spectra)):
        wn = sp.index.values.astype(float)
        ab = sp.values.astype(float)
        display_y = _to_display(ab, y_out)

        if i == 0:
            step = offset or float(np.max(display_y)) * 1.15

        shift = i * step
        color = COMPARE_PALETTE[i % len(COMPARE_PALETTE)]

        ax.plot(wn, display_y + shift, color=color, linewidth=1.0, label=name, zorder=5 + i)
        ax.axhline(shift, color=color, linewidth=0.4, linestyle="--", alpha=0.4)

        if annotate_first and i == 0:
            hits = _detect(sp, library, selector)
            top8 = hits[:8]
            y_base = float(display_y.max()) + shift + step * 0.08
            y_step = step * 0.18

            positions = _resolve_label_positions(top8, y_base=y_base, y_step=y_step, min_wn_gap=140.0)
            for lp in positions:
                h = lp["hit"]
                c = COMPARE_PALETTE[0]
                peak_display_y = _to_display(h.peak_absorbance, y_out) + shift
                ax.annotate(
                    h.rule.compound_class,
                    xy=(h.peak_wavenumber, peak_display_y),
                    xytext=(lp["x"], lp["y"]),
                    fontsize=5.5, ha="center", va="bottom", color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.7),
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=c, linewidth=0.5, alpha=0.9),
                    zorder=20,
                )

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(4100, 580)
    ax.set_xticks(range(4000, 500, -500))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100))
    ax.set_yticks([])
    ax.axvline(1500, color="#cccccc", linewidth=0.8, linestyle=":", zorder=0)

    ax.set_title(title or "ATR-FTIR Comparison Spectra", fontsize=12, pad=8, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
               edgecolor="#cccccc", title="Samples", title_fontsize=8)

    plt.tight_layout()
    out = out_path or "FTIR_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
