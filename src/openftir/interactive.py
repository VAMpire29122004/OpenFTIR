"""
Interactive spectrum viewer: builds a Plotly figure + an editable peaks
DataFrame from one sample. This is the data/rendering layer for
InteractiveEditor — Streamlit wiring (st.plotly_chart + st.data_editor)
is still the next checkpoint, not this one.

Detection is NOT reimplemented here. build_interactive_plot() pulls the
spectrum via profiler.get_spectrum() and classifies via profiler.classifier
— the exact same objects profile_sample() uses — so this can never drift
out of sync with what CompoundProfiler reports for the same sample.

DESIGN DECISION: when multiple reference-library rules claim the same
physical peak (common — many functional-group windows genuinely overlap,
e.g. the 1600-1700 cm-1 amide/alkene/ketone region), this module groups
them into one clustered marker/label on the plot rather than silently
picking a winner (that's what plotting.py's static-export NMS does).
Every candidate class is still listed as its own row in peaks_df, so the
researcher — who knows what they actually ran — decides via the Include
checkbox which candidate(s) apply. The algorithm surfaces the ambiguity;
it doesn't resolve it.

Clustering is ANCHORED (span from each cluster's first/lowest-wavenumber
member), not chained (span from the last-added member). Chaining lets
cluster width grow unboundedly as long as consecutive points are each
under min_dist apart — tested against real data, this merged an 11-class,
55 cm-1-wide span into one marker. Anchoring caps every cluster's total
width at min_dist by construction.

KNOWN LIMITATION: label placement is Plotly's default arrow-offset
annotation, not the collision-aware placement plotting.py uses for the
static matplotlib export. Dense clusters may still overlap visually.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .profiler import CompoundProfiler
from .models import WindowResult
from .clustering import cluster_points, cluster_label, cluster_tier

TIER_COLORS = {
    "very_strong": "#d62728",
    "strong"     : "#ff7f0e",
    "medium"     : "#2ca02c",
    "weak"       : "#9467bd",
    "manual"     : "#e377c2",  # distinct from detection tiers — flags
                                # peaks the user added by hand
}


PEAK_COLUMNS = ["Cluster", "Include", "Peak (cm⁻¹)", "Group", "Class",
                "Intensity", "Absorbance", "wn_low", "wn_high"]
RESOLVED_COLUMNS = ["Include", "Peak (cm⁻¹)", "Group", "Class",
                    "Intensity", "Absorbance", "wn_low", "wn_high"]


def _to_display(ab, y_out: str):
    if y_out == "trans_pct":
        return 100.0 * (10.0 ** -ab)
    if y_out == "trans_frac":
        return 10.0 ** -ab
    return ab


def _ylabel_for(y_out: str) -> str:
    if y_out == "trans_pct":
        return "Transmittance (%)"
    if y_out == "trans_frac":
        return "Transmittance (fraction)"
    return "Absorbance (a.u.)"


def _hit_to_point(h: WindowResult) -> dict:
    return {
        "wn"     : h.peak_wavenumber,
        "abs"    : h.peak_absorbance,
        "cls"    : h.rule.compound_class,
        "group"  : h.rule.group,
        "tier"   : h.rule.intensity_tier,
        "wn_low" : h.rule.wn_low,
        "wn_high": h.rule.wn_high,
    }


def build_peaks_dataframe(clusters: list) -> pd.DataFrame:
    """
    One row per candidate class, grouped by Cluster id for readability.
    'Include' is the editable column st.data_editor will expose — the
    researcher decides per-candidate which apply to their sample.
    NOTE: Cluster ids here are for human reference only (this detection
    pass). rebuild_figure_from_edits() re-clusters from scratch based on
    whatever rows are still checked, so edits never rely on a stale id.
    """
    rows = []
    for cluster_id, cluster in enumerate(sorted(
            clusters, key=lambda c: max(p["abs"] for p in c), reverse=True)):
        for p in sorted(cluster, key=lambda x: x["abs"], reverse=True):
            rows.append({
                "Cluster"      : cluster_id,
                "Include"      : True,
                "Peak (cm⁻¹)"  : round(p["wn"], 1),
                "Group"        : p["group"],
                "Class"        : p["cls"],
                "Intensity"    : p["tier"],
                "Absorbance"   : round(p["abs"], 4),
                "wn_low"       : p["wn_low"],
                "wn_high"      : p["wn_high"],
            })
    return pd.DataFrame(rows, columns=PEAK_COLUMNS)


def resolve_edited_peaks(edited_df: pd.DataFrame, spectrum: pd.Series,
                          absorbance_mode: str = "auto") -> pd.DataFrame:
    """
    Cleans up an edited peaks table — handles rows the user added by hand via
    st.data_editor's dynamic-row mode, which won't have Absorbance/Intensity
    filled in the way a detected hit does.

    absorbance_mode is a toggle, not a forced choice:
      - "auto":   any row missing Absorbance gets it looked up from the real
                  spectrum at the nearest wavenumber. A value the user DID
                  type is never overwritten, even in auto mode.
      - "manual": a row missing Absorbance is dropped rather than guessed.
    Rows missing a wavenumber or a class label are dropped either way.
    Rows missing Intensity default to the "manual" tier.
    """
    x = spectrum.index.values.astype(float)
    y = spectrum.values.astype(float)

    resolved_rows = []
    for _, row in edited_df.iterrows():
        wn = row.get("Peak (cm⁻¹)")
        if pd.isna(wn):
            continue

        cls = row.get("Class")
        if not cls or (isinstance(cls, float) and pd.isna(cls)):
            continue

        abs_val = row.get("Absorbance")
        if pd.isna(abs_val):
            if absorbance_mode == "auto":
                nearest_idx = int(np.argmin(np.abs(x - float(wn))))
                abs_val = float(y[nearest_idx])
            else:
                continue

        tier = row.get("Intensity")
        if not tier or (isinstance(tier, float) and pd.isna(tier)):
            tier = "manual"

        wn_low = row.get("wn_low")
        wn_high = row.get("wn_high")
        if pd.isna(wn_low) or pd.isna(wn_high):
            # Manually-added row (or bounds lost in the edit round-trip) --
            # same ±5 cm-1 convention ReferenceLibrary uses for single-value
            # CSV positions, so a synthetic rule still has a sane window.
            wn_low, wn_high = float(wn) - 5.0, float(wn) + 5.0

        inc = row.get("Include")
        include = True if (inc is None or pd.isna(inc)) else bool(inc)

        resolved_rows.append({
            "Include"      : include,
            "Peak (cm⁻¹)"  : float(wn),
            "Group"        : row.get("Group", "") or "",
            "Class"        : cls,
            "Intensity"    : tier,
            "Absorbance"   : float(abs_val),
            "wn_low"       : float(wn_low),
            "wn_high"      : float(wn_high),
        })
    return pd.DataFrame(resolved_rows, columns=RESOLVED_COLUMNS)


def resolved_peaks_to_window_results(resolved_df: pd.DataFrame) -> list:
    """
    Converts a resolved (post-edit) peaks table back into real WindowResult
    objects, so plotting.py's static export can render the researcher's
    edited peak list instead of running detection again. Each row gets a
    synthetic PeakRule built directly from its own data — peak_details is
    set to the tier keyword itself (e.g. "very_strong"), which works because
    PeakRule.intensity_tier normalizes underscore vs. space (see models.py).
    """
    from .models import PeakRule, WindowResult

    results = []
    for _, row in resolved_df.iterrows():
        rule = PeakRule(
            wn_low=float(row["wn_low"]),
            wn_high=float(row["wn_high"]),
            group=row.get("Group", "") or "",
            compound_class=row["Class"],
            peak_details=row["Intensity"],
        )
        results.append(WindowResult(
            rule=rule,
            peak_absorbance=float(row["Absorbance"]),
            mean_absorbance=float(row["Absorbance"]),  # not tracked post-edit; peak value is the only one that matters for rendering
            peak_wavenumber=float(row["Peak (cm⁻¹)"]),
            detected=True,  # by definition -- these are the rows the researcher kept
        ))
    return results


def _apply_layout(fig: go.Figure, sample_name: str, y_out: str):
    """Shared layout. Background is forced white (paper + plot area + black
    text) so the chart looks the same regardless of the viewer's light/dark
    theme -- pair with st.plotly_chart(theme=None), otherwise Streamlit's own
    theme overrides these colors."""
    axis_style = dict(showline=True, linecolor="#333333", gridcolor="#e6e6e6",
                      zerolinecolor="#e6e6e6", color="#111111")
    fig.update_layout(
        xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed", **axis_style),
        yaxis=dict(title=_ylabel_for(y_out), **axis_style),
        title=f"ATR-FTIR Spectrum — {sample_name}",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        height=550,
        showlegend=False,
    )


def _draw_clusters(fig: go.Figure, clusters: list, y_out: str):
    for cluster in clusters:
        color = TIER_COLORS.get(cluster_tier(cluster), "#888888")
        rep_wn = sum(p["wn"] for p in cluster) / len(cluster)
        rep_abs = max(p["abs"] for p in cluster)
        rep_display_y = _to_display(rep_abs, y_out)
        is_ambiguous = len(cluster) > 1
        label = cluster_label(cluster)

        fig.add_trace(go.Scatter(
            x=[rep_wn], y=[rep_display_y],
            mode="markers",
            marker=dict(color=color, size=9 if is_ambiguous else 8,
                        symbol="diamond-open" if is_ambiguous else "circle-open",
                        line=dict(width=2)),
            hovertemplate=f"{label}<br>%{{x:.0f}} cm⁻¹<extra></extra>",
            showlegend=False,
        ))
        fig.add_annotation(
            x=rep_wn, y=rep_display_y,
            text=f"{label}<br>{rep_wn:.0f} cm⁻¹",
            showarrow=True, arrowhead=2, arrowsize=0.8, arrowcolor=color,
            font=dict(size=10, color=color),
            bgcolor="white", bordercolor=color,
            borderwidth=2 if is_ambiguous else 1,
            ax=0, ay=-40,
        )


def build_interactive_plot(sample_name: str, profiler: CompoundProfiler,
                            y_out: str = "abs", max_labels: int = 15,
                            cluster_dist: float = 30.0):
    """
    Returns (fig, peaks_df).
    Uses profiler.get_spectrum() so the plotted curve is EXACTLY what
    detection ran against — preprocessed if profiler.preprocessor is set.
    """
    spectrum = profiler.get_spectrum(sample_name)
    wn = spectrum.index.values.astype(float)
    ab = spectrum.values.astype(float)
    display_y = _to_display(ab, y_out)

    hits: list = profiler.classifier.classify(spectrum)
    points = [_hit_to_point(h) for h in hits]
    clusters = cluster_points(points, min_dist=cluster_dist)
    clusters.sort(key=lambda c: max(p["abs"] for p in c), reverse=True)
    top_clusters = clusters[:max_labels]

    peaks_df = build_peaks_dataframe(top_clusters)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wn, y=display_y, mode="lines",
        line=dict(color="#1a1a2e", width=1.3),
        name=sample_name,
        hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.4f}<extra></extra>",
    ))

    _draw_clusters(fig, top_clusters, y_out)

    _apply_layout(fig, sample_name, y_out)

    return fig, peaks_df


def rebuild_figure_from_edits(sample_name: str, profiler: CompoundProfiler,
                               edited_peaks_df: pd.DataFrame, y_out: str = "abs",
                               absorbance_mode: str = "auto",
                               cluster_dist: float = 30.0):
    """
    Re-renders the spectrum curve + only the rows the user left checked in
    'Include'. Re-clusters from scratch on every call — so unchecking a
    cluster down to one candidate collapses it to a clean single label, and
    a manually-added peak merges into a nearby cluster if it's actually
    close, rather than trusting a Cluster id that may now be stale.
    """
    spectrum = profiler.get_spectrum(sample_name)
    wn = spectrum.index.values.astype(float)
    ab = spectrum.values.astype(float)
    display_y = _to_display(ab, y_out)

    resolved = resolve_edited_peaks(edited_peaks_df, spectrum, absorbance_mode)
    kept = resolved[resolved["Include"].astype(bool)] if not resolved.empty else resolved

    points = [{
        "wn"   : float(row["Peak (cm⁻¹)"]),
        "abs"  : float(row["Absorbance"]),
        "cls"  : row["Class"],
        "group": row.get("Group", ""),
        "tier" : row["Intensity"],
    } for _, row in kept.iterrows()]

    clusters = cluster_points(points, min_dist=cluster_dist)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wn, y=display_y, mode="lines",
        line=dict(color="#1a1a2e", width=1.3),
        name=sample_name,
        hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.4f}<extra></extra>",
    ))

    _draw_clusters(fig, clusters, y_out)

    _apply_layout(fig, sample_name, y_out)
    return fig
