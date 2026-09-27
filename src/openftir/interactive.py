"""
Interactive spectrum viewer: builds a Plotly figure + an editable peaks
DataFrame from one sample. This is the data/rendering layer for
InteractiveEditor — wiring it into Streamlit (st.plotly_chart +
st.data_editor, and regenerating the figure when the user edits a row)
is the next checkpoint, not this one.

Detection is NOT reimplemented here. build_interactive_plot() pulls the
spectrum via profiler.get_spectrum() and classifies via profiler.classifier
— the exact same objects profile_sample() uses — so this can never drift
out of sync with what CompoundProfiler reports for the same sample.

KNOWN LIMITATION: label placement here is Plotly's default arrow-offset
annotation (ax/ay pixel offset from each point), not the collision-aware
placement plotting.py uses for the static matplotlib export. Plotly
annotations don't take plot-coordinate offsets the way _resolve_label_positions
computes them, so dense clusters of peaks may still overlap in this view.
Flagging as a gap to revisit, not silently working around it.
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
    "manual"     : "#e377c2",  # distinct from the four detection tiers —
                                # visually flags peaks the user added by hand,
                                # not the algorithm
}


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


def resolve_edited_peaks(edited_df: pd.DataFrame, spectrum: pd.Series,
                          absorbance_mode: str = "auto") -> pd.DataFrame:
    """
    Cleans up an edited peaks table — handles rows the user added by hand via
    st.data_editor's dynamic-row mode, which won't have Absorbance/Intensity
    filled in the way a detected hit does.

    absorbance_mode is a toggle, not a forced choice (per user request):
      - "auto":   any row missing an Absorbance value gets it looked up from
                  the actual spectrum at the nearest wavenumber. A value the
                  user DID type is never overwritten, even in auto mode.
      - "manual": a row missing Absorbance is dropped rather than guessed —
                  there's nothing meaningful to plot without a y-value.
    Rows missing a wavenumber or a class label are dropped either way.
    Rows missing Intensity default to the "manual" tier (its own marker
    color), distinguishing user-asserted peaks from algorithm detections.
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
                continue  # manual mode: no typed value, nothing to plot

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


def build_peaks_dataframe(hits: list) -> pd.DataFrame:
    """
    One row per detected hit. 'Include' is the editable column st.data_editor
    will expose — unchecking a row is meant to drop a false-positive
    detection from the rendered figure, without touching the underlying
    WindowResult objects or re-running classification.
    """
    rows = []
    for h in sorted(hits, key=lambda w: w.peak_absorbance, reverse=True):
        rows.append({
            "Include"      : True,
            "Peak (cm⁻¹)"  : round(h.peak_wavenumber, 1),
            "Group"        : h.rule.group,
            "Class"        : h.rule.compound_class,
            "Intensity"    : h.rule.intensity_tier,
            "Absorbance"   : round(h.peak_absorbance, 4),
        })
    return pd.DataFrame(rows)


def build_interactive_plot(sample_name: str, profiler: CompoundProfiler,
                            y_out: str = "abs", max_labels: int = 15):
    """
    Returns (fig, peaks_df).
    Uses profiler.get_spectrum() so the plotted curve is EXACTLY what
    detection ran against — preprocessed if profiler.preprocessor is set,
    raw otherwise.
    """
    spectrum = profiler.get_spectrum(sample_name)
    wn = spectrum.index.values.astype(float)
    ab = spectrum.values.astype(float)
    display_y = _to_display(ab, y_out)

    hits: list[WindowResult] = profiler.classifier.classify(spectrum)
    hits.sort(key=lambda w: w.peak_absorbance, reverse=True)
    top = hits[:max_labels]

    peaks_df = build_peaks_dataframe(top)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wn, y=display_y, mode="lines",
        line=dict(color="#1a1a2e", width=1.3),
        name=sample_name,
        hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.4f}<extra></extra>",
    ))

    for h in top:
        color = TIER_COLORS.get(h.rule.intensity_tier, "#888888")
        peak_display_y = _to_display(h.peak_absorbance, y_out)

        fig.add_trace(go.Scatter(
            x=[h.peak_wavenumber], y=[peak_display_y],
            mode="markers",
            marker=dict(color=color, size=8, symbol="circle-open", line=dict(width=2)),
            hovertemplate=f"{h.rule.compound_class}<br>{h.rule.group}<br>"
                          "%{x:.0f} cm⁻¹<extra></extra>",
            showlegend=False,
        ))
        fig.add_annotation(
            x=h.peak_wavenumber, y=peak_display_y,
            text=f"{h.rule.compound_class}<br>{h.peak_wavenumber:.0f} cm⁻¹",
            showarrow=True, arrowhead=2, arrowsize=0.8, arrowcolor=color,
            font=dict(size=10, color=color),
            bgcolor="white", bordercolor=color, borderwidth=1,
            ax=0, ay=-40,
        )

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
                               absorbance_mode: str = "auto"):
    """
    Re-renders the spectrum curve + only the rows the user left checked in
    'Include'. Runs edited_peaks_df through resolve_edited_peaks() first, so
    manually-added rows (blank Absorbance/Intensity from st.data_editor's
    dynamic-row mode) get handled per absorbance_mode before anything is
    plotted. Peak metadata is taken from the resolved DataFrame, not
    re-derived from WindowResult objects — so a user's manual correction is
    what gets drawn, not the original detection.
    """
    spectrum = profiler.get_spectrum(sample_name)
    wn = spectrum.index.values.astype(float)
    ab = spectrum.values.astype(float)
    display_y = _to_display(ab, y_out)

    resolved = resolve_edited_peaks(edited_peaks_df, spectrum, absorbance_mode)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wn, y=display_y, mode="lines",
        line=dict(color="#1a1a2e", width=1.3),
        name=sample_name,
        hovertemplate="%{x:.0f} cm⁻¹<br>%{y:.4f}<extra></extra>",
    ))

    kept = resolved[resolved["Include"]] if not resolved.empty else resolved
    for _, row in kept.iterrows():
        color = TIER_COLORS.get(row["Intensity"], "#888888")
        peak_wn = float(row["Peak (cm⁻¹)"])
        peak_display_y = _to_display(float(row["Absorbance"]), y_out)

        fig.add_trace(go.Scatter(
            x=[peak_wn], y=[peak_display_y],
            mode="markers",
            marker=dict(color=color, size=8, symbol="circle-open", line=dict(width=2)),
            showlegend=False,
        ))
        fig.add_annotation(
            x=peak_wn, y=peak_display_y,
            text=f"{row['Class']}<br>{peak_wn:.0f} cm⁻¹",
            showarrow=True, arrowhead=2, arrowsize=0.8, arrowcolor=color,
            font=dict(size=10, color=color),
            bgcolor="white", bordercolor=color, borderwidth=1,
            ax=0, ay=-40,
        )

    fig.update_layout(
        xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed"),
        yaxis=dict(title=_ylabel_for(y_out)),
        title=f"ATR-FTIR Spectrum — {sample_name}",
        template="plotly_white",
        height=550,
        showlegend=False,
    )
    return fig
