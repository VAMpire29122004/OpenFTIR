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

TIER_COLORS = {
    "very_strong": "#d62728",
    "strong"     : "#ff7f0e",
    "medium"     : "#2ca02c",
    "weak"       : "#9467bd",
    "manual"     : "#e377c2",  # distinct from detection tiers — flags
                                # peaks the user added by hand
}

# Priority order for picking a cluster's marker color when its members span
# multiple tiers — highest-confidence tier wins the color.
_TIER_PRIORITY = ["very_strong", "strong", "medium", "weak", "manual"]


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
        "wn"   : h.peak_wavenumber,
        "abs"  : h.peak_absorbance,
        "cls"  : h.rule.compound_class,
        "group": h.rule.group,
        "tier" : h.rule.intensity_tier,
    }


def _cluster_points(points: list, min_dist: float = 30.0) -> list:
    """
    Groups points whose wavenumbers fall within min_dist of their cluster's
    FIRST (lowest-wavenumber) member — anchored, not chained, so every
    cluster's total span is guaranteed < min_dist. See module docstring for
    why chaining was rejected (unbounded cluster width on real data).
    """
    if not points:
        return []
    ordered = sorted(points, key=lambda p: p["wn"])
    clusters = [[ordered[0]]]
    for p in ordered[1:]:
        anchor = clusters[-1][0]["wn"]
        if p["wn"] - anchor < min_dist:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def _cluster_label(cluster: list, max_names: int = 4) -> str:
    names = sorted({p["cls"] for p in cluster})
    if len(names) > max_names:
        return ", ".join(names[:max_names]) + f" (+{len(names) - max_names} more)"
    return " / ".join(names)


def _cluster_color(cluster: list) -> str:
    tiers_present = {p["tier"] for p in cluster}
    for tier in _TIER_PRIORITY:
        if tier in tiers_present:
            return TIER_COLORS.get(tier, "#888888")
    return "#888888"


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
            })
    return pd.DataFrame(rows)


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

        resolved_rows.append({
            "Include"      : row.get("Include", True),
            "Peak (cm⁻¹)"  : float(wn),
            "Group"        : row.get("Group", "") or "",
            "Class"        : cls,
            "Intensity"    : tier,
            "Absorbance"   : float(abs_val),
        })
    return pd.DataFrame(resolved_rows)


def _draw_clusters(fig: go.Figure, clusters: list, y_out: str):
    for cluster in clusters:
        color = _cluster_color(cluster)
        rep_wn = sum(p["wn"] for p in cluster) / len(cluster)
        rep_abs = max(p["abs"] for p in cluster)
        rep_display_y = _to_display(rep_abs, y_out)
        is_ambiguous = len(cluster) > 1
        label = _cluster_label(cluster)

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
    clusters = _cluster_points(points, min_dist=cluster_dist)
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

    fig.update_layout(
        xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed"),
        yaxis=dict(title=_ylabel_for(y_out)),
        title=f"ATR-FTIR Spectrum — {sample_name}",
        template="plotly_white",
        height=550,
        showlegend=False,
    )

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
    kept = resolved[resolved["Include"]] if not resolved.empty else resolved

    points = [{
        "wn"   : float(row["Peak (cm⁻¹)"]),
        "abs"  : float(row["Absorbance"]),
        "cls"  : row["Class"],
        "group": row.get("Group", ""),
        "tier" : row["Intensity"],
    } for _, row in kept.iterrows()]

    clusters = _cluster_points(points, min_dist=cluster_dist)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wn, y=display_y, mode="lines",
        line=dict(color="#1a1a2e", width=1.3),
        name=sample_name,
        hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.4f}<extra></extra>",
    ))

    _draw_clusters(fig, clusters, y_out)

    fig.update_layout(
        xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed"),
        yaxis=dict(title=_ylabel_for(y_out)),
        title=f"ATR-FTIR Spectrum — {sample_name}",
        template="plotly_white",
        height=550,
        showlegend=False,
    )
    return fig
