import streamlit as st
import pandas as pd

from openftir import ReferenceLibrary, standardize_to_absorbance, plot_single, plot_compare

st.set_page_config(page_title="OpenFTIR", page_icon="🔬", layout="wide")
st.title("🔬 OpenFTIR: Context-Aware Spectrum Analyzer")

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

        st.header("Plot Generation")
        mode = st.radio("Select Mode", ["Single Spectrum", "Compare Spectra"], horizontal=True)
        available_samples = df_abs.index.tolist()

        if mode == "Single Spectrum":
            selected_sample = st.selectbox("Select Sample", available_samples)

            if st.button("Generate Plot", type="primary"):
                with st.spinner("Plotting..."):
                    out_path = plot_single(selected_sample, df_abs, library, y_out=y_out)
                    st.image(out_path, use_container_width=True)

        else:
            selected_samples = st.multiselect("Select Samples to Compare", available_samples)

            if st.button("Generate Comparison Plot", type="primary"):
                if len(selected_samples) < 2:
                    st.warning("Please select at least two samples to compare.")
                else:
                    with st.spinner("Plotting..."):
                        out_path = plot_compare(selected_samples, df_abs, library, y_out=y_out)
                        st.image(out_path, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing data: {e}. Please check your CSV formats.")
else:
    st.info("👈 Please upload both your Spectra and Reference CSV files in the sidebar to begin.")
