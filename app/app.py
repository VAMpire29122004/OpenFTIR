import hashlib
import os
import tempfile

import pandas as pd
import streamlit as st

from openftir import ReferenceLibrary, standardize_to_absorbance, plot_single, plot_compare
from openftir.profiler import CompoundProfiler
from openftir.selector import FeatureSelector, DeconvolutionSelector
from openftir.preprocessing import SpectrumPreprocessor
from openftir.interactive import (
    build_interactive_plot,
    rebuild_figure_from_edits,
    resolve_edited_peaks,
    resolved_peaks_to_window_results,
)

st.set_page_config(page_title="OpenFTIR", page_icon="🔬", layout="wide")
st.title("🔬 OpenFTIR: Context-Aware Spectrum Analyzer")

# Solid, opaque white behind every Plotly chart -- independent of the viewer's
# light/dark theme, so the plot can never look transparent or washed out.
st.markdown(
    """
    <style>
    [data-testid="stPlotlyChart"] {
        background-color: #FFFFFF !important;
        border-radius: 6px;
        padding: 8px;
    }
    [data-testid="stPlotlyChart"] .main-svg {
        background-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: Data Upload ---
st.sidebar.header("1. Data Upload")
spectra_file = st.sidebar.file_uploader("Upload Spectra CSV", type=["csv"])
ref_file = st.sidebar.file_uploader("Upload Reference CSV", type=["csv"])

with st.sidebar.expander("ℹ️ View Expected CSV Formats"):
    st.markdown("**Spectra File (Standard)**")
    st.markdown("*(Wavenumbers as columns, Samples as rows)*")
    st.markdown("""
| Sample_Name | 4000 | 3999 | 3998 | ... |
| :--- | :--- | :--- | :--- | :--- |
| Sample_A | 0.05 | 0.06 | 0.05 | ... |
| Sample_B | 0.10 | 0.11 | 0.10 | ... |
    """)

    st.markdown("**Spectra File (Transposed)**")
    st.markdown("*(Wavenumbers as rows, Samples as columns. Check the 'Transpose' box below!)*")
    st.markdown("""
| Lambda | Sample_A | Sample_B |
| :--- | :--- | :--- |
| 4000 | 0.05 | 0.10 |
| 3999 | 0.06 | 0.11 |
| 3998 | 0.05 | 0.10 |
    """)

    st.markdown("**Reference Peak File**")
    st.markdown("*(Must contain these column headers)*")
    st.markdown("""
| Peak Position | Group | Class | Peak Details |
| :--- | :--- | :--- | :--- |
| 3200-3550 | O-H stretching | alcohol | strong, broad |
| 1700 | C=O stretching | ketone | strong, sharp |
    """)

st.sidebar.header("2. Settings")
transpose = st.sidebar.checkbox("Transpose Data (Wavenumbers as rows)", value=False)
y_in = st.sidebar.selectbox("Input Data Format", ["auto", "abs", "trans_frac", "trans_pct"])
y_out = st.sidebar.selectbox("Output Plot Format", ["abs", "trans_frac", "trans_pct"])

TIER_OPTIONS = ["very_strong", "strong", "medium", "weak", "manual"]

# --- Main Dashboard ---
if spectra_file and ref_file:
    try:
        if transpose:
            df_raw = pd.read_csv(spectra_file, index_col=0).T
        else:
            df_raw = pd.read_csv(spectra_file).set_index("Sample_Name")
        df_raw.columns = df_raw.columns.astype(float)

        library = ReferenceLibrary(ref_file)
        df_abs = standardize_to_absorbance(df_raw, mode=y_in)

        st.success(f"✅ Data loaded successfully! Found {len(df_abs)} samples, {len(library)} reference rules.")

        mode = st.radio("Select Mode", ["Interactive Spectrum", "Compare Spectra"], horizontal=True)
        available_samples = df_abs.index.tolist()

        # =====================================================================
        # Interactive Spectrum (replaces the old static Single Spectrum mode)
        # =====================================================================
        if mode == "Interactive Spectrum":
            selected_sample = st.selectbox("Select Sample", available_samples)

            c1, c2, c3 = st.columns(3)
            with c1:
                engine = st.selectbox(
                    "Detection engine",
                    ["Rule-based (fast)", "Deconvolution (Gaussian fit)"],
                    help="Deconvolution separates overlapping bands and rejects shoulders of "
                         "neighbouring peaks, but its prominence cut-off can miss subtle real "
                         "peaks near strong ones — add those by hand below.",
                )
            with c2:
                smooth = st.checkbox("Savitzky-Golay smoothing", value=False)
            with c3:
                baseline = st.checkbox(
                    "ALS baseline correction", value=False,
                    help="Caution: the intensity thresholds were calibrated on uncorrected "
                         "absorbance. Baseline correction shifts the scale and can suppress "
                         "weak-tier detections.",
                )

            abs_choice = st.radio(
                "Absorbance for peaks you add by hand",
                ["Auto-lookup from spectrum", "Manual entry"], horizontal=True,
            )
            abs_mode = "auto" if abs_choice.startswith("Auto") else "manual"

            selector = DeconvolutionSelector() if engine.startswith("Deconv") else FeatureSelector()
            preprocessor = (SpectrumPreprocessor(smooth=smooth, baseline_correct=baseline)
                            if (smooth or baseline) else None)
            profiler = CompoundProfiler(library, selector=selector, preprocessor=preprocessor)
            profiler.spectra_df = df_abs

            # Re-detect only when something that affects detection changes —
            # NOT on every table edit (that would throw the researcher's edits away).
            sig = (spectra_file.name, spectra_file.size, ref_file.name, ref_file.size,
                   transpose, y_in, selected_sample, engine, smooth, baseline)
            sig_hash = hashlib.md5(repr(sig).encode()).hexdigest()[:10]
            if st.session_state.get("peaks_sig") != sig:
                with st.spinner("Detecting peaks..."):
                    _, base_df = build_interactive_plot(selected_sample, profiler, y_out=y_out)
                st.session_state["peaks_sig"] = sig
                st.session_state["peaks_base"] = base_df
                st.session_state.pop("pub_fig", None)
            base_df = st.session_state["peaks_base"]

            plot_area = st.container()

            st.markdown("#### Peak table")
            st.caption(
                "Uncheck **Include** to drop a candidate. Peaks that claim the same position share a "
                "**Cluster** and appear as one diamond marker on the plot — uncheck the ones that "
                "don't apply to your sample. Add a missed peak with the empty row at the bottom "
                "(needs a wavenumber and a class name)."
            )
            edited = st.data_editor(
                base_df,
                key=f"editor_{sig_hash}",
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "Cluster": st.column_config.NumberColumn("Cluster", disabled=True, format="%d"),
                    "Include": st.column_config.CheckboxColumn("Include", default=True),
                    "Peak (cm⁻¹)": st.column_config.NumberColumn("Peak (cm⁻¹)", format="%.1f"),
                    "Group": st.column_config.TextColumn("Group"),
                    "Class": st.column_config.TextColumn("Class"),
                    "Intensity": st.column_config.SelectboxColumn("Intensity", options=TIER_OPTIONS),
                    "Absorbance": st.column_config.NumberColumn(
                        "Absorbance", format="%.4f",
                        help="Leave blank on a new row to auto-look-up from the spectrum "
                             "(when Auto-lookup is selected)."),
                    "wn_low": None,   # bookkeeping columns, hidden from the researcher
                    "wn_high": None,
                },
            )

            fig = rebuild_figure_from_edits(selected_sample, profiler, edited,
                                            y_out=y_out, absorbance_mode=abs_mode)
            with plot_area:
                st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})

            # ---- Publication export: renders the EDITED table, not a fresh detection ----
            st.markdown("#### Publication figure")
            spectrum = profiler.get_spectrum(selected_sample)
            resolved = resolve_edited_peaks(edited, spectrum, abs_mode)
            kept = resolved[resolved["Include"].astype(bool)] if not resolved.empty else resolved
            kept_csv = kept.drop(columns=["wn_low", "wn_high"]).to_csv(index=False)

            if st.button("Generate publication figure", type="primary"):
                with st.spinner("Rendering..."):
                    one = pd.DataFrame([spectrum.values], index=[selected_sample], columns=spectrum.index)
                    with tempfile.TemporaryDirectory() as d:
                        out = plot_single(
                            selected_sample, one, library=None,
                            out_path=os.path.join(d, "fig.png"),
                            hits_override=resolved_peaks_to_window_results(kept),
                            max_labels=20, y_out=y_out,
                        )
                        with open(out, "rb") as f:
                            png_bytes = f.read()
                st.session_state["pub_fig"] = {"sig": sig, "png": png_bytes, "csv": kept_csv}

            pub = st.session_state.get("pub_fig")
            if pub and pub["sig"] == sig:
                if pub["csv"] != kept_csv:
                    st.warning("The peak table has changed since this figure was generated — "
                               "click **Generate publication figure** again to update it.")
                st.image(pub["png"], width="stretch")
                d1, d2 = st.columns(2)
                safe = selected_sample.replace(" ", "_").replace("/", "_")
                d1.download_button("⬇️ Download figure (PNG)", pub["png"],
                                   file_name=f"{safe}_FTIR_edited.png", mime="image/png")
                d2.download_button("⬇️ Download peak table (CSV)", pub["csv"],
                                   file_name=f"{safe}_peaks.csv", mime="text/csv")

        # =====================================================================
        # Compare Spectra (unchanged behaviour: static stacked plot)
        # =====================================================================
        else:
            selected_samples = st.multiselect("Select Samples to Compare", available_samples)

            if st.button("Generate Comparison Plot", type="primary"):
                if len(selected_samples) < 2:
                    st.warning("Please select at least two samples to compare.")
                else:
                    with st.spinner("Plotting..."):
                        with tempfile.TemporaryDirectory() as d:
                            out = plot_compare(selected_samples, df_abs, library, y_out=y_out,
                                               out_path=os.path.join(d, "compare.png"))
                            with open(out, "rb") as f:
                                png_bytes = f.read()
                    st.image(png_bytes, width="stretch")
                    st.download_button("⬇️ Download figure (PNG)", png_bytes,
                                       file_name="FTIR_comparison.png", mime="image/png")

    except Exception as e:
        st.error(f"Error processing data: {e}. Please check your CSV formats.")
else:
    st.info("👈 Please upload both your Spectra and Reference CSV files in the sidebar to begin.")
