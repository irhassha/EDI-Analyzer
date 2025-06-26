import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="EDI Forecaster", layout="wide")

# Fungsi utilitas AgGrid
def display_aggrid_centered(df, fit_columns=True):
    if df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(cellStyle={'textAlign': 'center'}, wrapText=True, autoHeight=True)
    grid_options = gb.build()
    AgGrid(df, gridOptions=grid_options, fit_columns_on_grid_load=fit_columns, theme="streamlit")

# --- Fungsi-fungsi utama seperti parse_edi_to_pivot, forecast_next_value_wma, compare_multiple_pivots, dst. tetap sama ---
# Kamu bisa tempel ulang semua fungsi dari script sebelumnya di sini TANPA DIUBAH, karena sudah berfungsi dengan baik.
# (sengaja tidak ditampilkan ulang untuk menghemat ruang canvas ini)

# --- LAYOUT STREAMLIT ---
st.title("🚢 EDI File Comparator & Forecaster")
st.caption("Upload EDI files to compare and forecast the load for the next vessel.")

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    uploaded_files = st.file_uploader("1. Upload your .EDI files here", type=["edi", "txt"], accept_multiple_files=True)

    uploaded_file_buffers = {}
    if uploaded_files:
        # Simpan semua file ke buffer
        uploaded_file_buffers = {f.name: io.BytesIO(f.read()) for f in uploaded_files}
        file_names = list(uploaded_file_buffers.keys())
        selected_files = st.multiselect("2. Select files (in order):", options=file_names, default=file_names)

        all_pods = []
        if selected_files:
            try:
                pivots_for_pods = {}
                for f_name in selected_files:
                    buffer = uploaded_file_buffers[f_name]
                    buffer.seek(0)
                    pivots_for_pods[f_name] = parse_edi_to_pivot(buffer)
                all_pods = sorted(list(pd.concat([df['Port of Discharge'] for df in pivots_for_pods.values() if not df.empty]).unique()))
            except Exception as e:
                st.error(f"Error getting PODs: {e}")

        excluded_pods = st.multiselect("3. Exclude Ports of Discharge (optional):", options=all_pods)

        num_clusters = st.number_input("4. Select number of clusters:", min_value=2, max_value=20, value=6, step=1,
                                       help="This will group the Bays into the selected number of ranges for analysis.")

if not uploaded_files or len(uploaded_files) < 2:
    st.info("ℹ️ Please upload at least 2 files in the sidebar to start the analysis.")
elif 'selected_files' in locals() and len(selected_files) < 2:
    st.warning("Please select at least 2 files in the sidebar to display the comparison.")
else:
    with st.spinner("Analyzing files..."):
        pivots_dict = {}
        for f_name in selected_files:
            buffer = uploaded_file_buffers[f_name]
            buffer.seek(0)
            pivots_dict[f_name] = parse_edi_to_pivot(buffer)

        comparison_df = compare_multiple_pivots(pivots_dict)

        if excluded_pods:
            comparison_df = comparison_df[~comparison_df['Port of Discharge'].isin(excluded_pods)]

    if comparison_df.empty:
        st.error("Could not generate a valid comparison from the selected files. Please check the files or your settings.")
    else:
        st.header("📊 Summary per Vessel")
        summary_table = create_summary_table(comparison_df)
        if not summary_table.empty:
            create_summary_chart(summary_table)
        else:
            st.warning("No valid data to create a summary.")

        st.markdown("---")
        st.header("🎯 Cluster Analysis")
        df_with_clusters = add_cluster_info(comparison_df, num_clusters)

        st.subheader("Forecast Allocation Summary per Cluster (in Boxes)")
        cluster_table = create_summarized_cluster_table(df_with_clusters)
        if not cluster_table.empty:
            display_aggrid_centered(cluster_table)

        st.subheader("Macro Slot Needs")
        macro_slot_table = create_macro_slot_table(df_with_clusters)
        if not macro_slot_table.empty:
            display_aggrid_centered(macro_slot_table.set_index('CLUSTER').reset_index())

        st.markdown("---")

        st.header("📥 Export Results")
        if not cluster_table.empty and not macro_slot_table.empty:
            export_data = {
                "Forecast Allocation": cluster_table,
                "Macro Slot Needs": macro_slot_table,
                "Detailed Data": comparison_df
            }
            generate_excel_download_link(export_data)

        st.markdown("---")

        with st.expander("Show Detailed Comparison & Forecast Table"):
            display_cols = [col for col in comparison_df.columns if not col.startswith('Weight')]
            display_aggrid_centered(comparison_df[display_cols])

        st.header("⚖️ Forecast Weight (VGM) Chart per Bay")
        create_colored_weight_chart(df_with_clusters)
