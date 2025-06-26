import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder

# Page config
st.set_page_config(page_title="EDI Forecaster", layout="wide")

# --- CUSTOM FUNCTION: CENTERED AGGRID WITH AUTO HEIGHT ---
def display_aggrid_centered(df, fit_columns=True, max_height=600, row_height=35):
    if df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(cellStyle={'textAlign': 'center'}, headerClass='centered-header')
    grid_options = gb.build()

    st.markdown("""
        <style>
        .ag-theme-streamlit .ag-header-cell-label {
            justify-content: center;
        }
        .ag-theme-streamlit .ag-header-cell {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    estimated_height = min(max_height, (len(df) + 1) * row_height)

    AgGrid(df,
           gridOptions=grid_options,
           fit_columns_on_grid_load=fit_columns,
           theme="streamlit",
           height=estimated_height)

# --- FUNGSI PIVOT EDI ---
def parse_edi_to_pivot(uploaded_file):
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8")
        lines = content.strip().splitlines()
    except Exception:
        return pd.DataFrame()

    all_records = []
    current_container_data = {}
    for line in lines:
        line = line.strip()
        if line.startswith("LOC+147+"):
            if current_container_data:
                all_records.append(current_container_data)
            current_container_data = {'Weight': 0}
            try:
                full_bay = line.split("+")[2].split(":")[0]
                current_container_data["Bay"] = full_bay[1:3] if len(full_bay) >= 3 else full_bay
            except IndexError:
                current_container_data["Bay"] = None
        elif line.startswith("LOC+11+"):
            try:
                pod = line.split("+")[2].split(":")[0]
                current_container_data["Port of Discharge"] = pod.replace("'", "")
            except IndexError:
                pass
        elif line.startswith("LOC+9+"):
            try:
                pol = line.split("+")[2].split(":")[0]
                current_container_data["Port of Loading"] = pol.strip().upper().replace("'", "")
            except IndexError:
                pass
        elif line.startswith("MEA+VGM++KGM:"):
            try:
                weight_str = line.split(':')[-1].replace("'", "").strip()
                current_container_data['Weight'] = pd.to_numeric(weight_str, errors='coerce')
            except (IndexError, ValueError):
                pass

    if current_container_data:
        all_records.append(current_container_data)

    if not all_records:
        return pd.DataFrame()

    df_all = pd.DataFrame(all_records)
    if "Port of Loading" not in df_all.columns:
        return pd.DataFrame()
    df_filtered = df_all.loc[df_all["Port of Loading"] == "IDJKT"].copy()
    if df_filtered.empty: return pd.DataFrame()
    df_display = df_filtered.drop(columns=["Port of Loading"]).dropna(subset=["Port of Discharge", "Bay"])
    df_display['Weight'] = df_display['Weight'].fillna(0)

    pivot_df = df_display.groupby(["Bay", "Port of Discharge"], as_index=False).agg(
        **{'Container Count': ('Bay', 'size'), 'Total Weight': ('Weight', 'sum')}
    )
    pivot_df['Bay'] = pd.to_numeric(pivot_df['Bay'], errors='coerce')
    pivot_df.dropna(subset=['Bay'], inplace=True)
    pivot_df['Bay'] = pivot_df['Bay'].astype(int)
    return pivot_df

def forecast_next_value_wma(series):
    if len(series.dropna()) < 2:
        return round(series.mean()) if not series.empty else 0
    y = series.values
    weights = np.arange(1, len(y) + 1)
    return max(0, round(np.average(y, weights=weights)))

def compare_multiple_pivots(pivots_dict_selected):
    dfs_to_merge = []
    for name, df in pivots_dict_selected.items():
        if not df.empty:
            renamed_df = df.set_index(["Bay", "Port of Discharge"]).rename(columns={
                "Container Count": f"Count ({name})",
                "Total Weight": f"Weight ({name})"
            })
            dfs_to_merge.append(renamed_df)
    if not dfs_to_merge: return pd.DataFrame()

    merged_df = pd.concat(dfs_to_merge, axis=1, join='outer').fillna(0).astype(int)
    count_cols = [col for col in merged_df.columns if col.startswith('Count')]
    weight_cols = [col for col in merged_df.columns if col.startswith('Weight')]
    if count_cols:
        merged_df['Forecast (Next Vessel)'] = merged_df[count_cols].apply(forecast_next_value_wma, axis=1).astype(int)
    if weight_cols:
        merged_df['Forecast Weight (KGM)'] = merged_df[weight_cols].apply(forecast_next_value_wma, axis=1).astype(int)
    return merged_df.reset_index().sort_values(by="Bay").reset_index(drop=True)

def create_summary_table(df):
    if df.empty: return pd.DataFrame()
    cols = [col for col in df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    return df[['Port of Discharge'] + cols].groupby('Port of Discharge').sum().reset_index()

def add_cluster_info(df, num_clusters=6):
    if df.empty or df['Bay'].nunique() < num_clusters:
        return df.assign(**{'Cluster ID': 0, 'Bay Range': 'N/A'})
    df['Cluster ID'] = pd.qcut(df['Bay'], q=num_clusters, labels=False, duplicates='drop')
    df['Bay Range'] = df.groupby('Cluster ID')['Bay'].transform(lambda x: f"{x.min()}-{x.max()}")
    return df

def create_summarized_cluster_table(df):
    if df.empty: return pd.DataFrame()
    df = df[df['Forecast (Next Vessel)'] > 0].copy()
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    pivot = df.pivot_table(index=['Bay Range'], columns='Allocation Column',
                           values='Forecast (Next Vessel)', aggfunc='sum', fill_value=0)
    pivot['_sort'] = pivot.index.str.split('-').str[0].astype(int)
    pivot = pivot.sort_values(by='_sort').drop(columns='_sort')
    pivot['Total Boxes'] = pivot.sum(axis=1)
    pivot = pivot.reset_index().rename(columns={'Bay Range': 'BAY'})
    pivot.insert(0, 'CLUSTER', range(1, len(pivot) + 1))
    return pivot

def create_macro_slot_table(df):
    if df.empty: return pd.DataFrame()
    df = df[df['Forecast (Next Vessel)'] > 0].copy()
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    pivot = df.pivot_table(index=['Bay Range'], columns='Allocation Column',
                           values='Forecast (Next Vessel)', aggfunc='sum', fill_value=0)
    for col in pivot.columns:
        if ' 20' in col:
            pivot[col] = np.ceil(pivot[col] / 30)
        elif ' 40' in col:
            pivot[col] = np.ceil(pivot[col] / 30) * 2
    pivot['Total Slot Needs'] = pivot.sum(axis=1)
    pivot = pivot.astype(int)
    pivot['_sort'] = pivot.index.str.split('-').str[0].astype(int)
    pivot = pivot.sort_values(by='_sort').drop(columns='_sort')
    pivot = pivot.reset_index().drop(columns='Bay Range')
    pivot.insert(0, 'CLUSTER', range(1, len(pivot) + 1))
    total = pivot.pop('Total Slot Needs')
    pivot.insert(1, 'Total Slot Needs', total)
    return pivot

def generate_excel_download_link(tables_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in tables_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    st.download_button(
        label="📥 Export All Tables to Excel",
        data=output.getvalue(),
        file_name="edi_analysis_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- STREAMLIT APP ---
st.title("🚢 EDI File Comparator & Forecaster")
st.caption("Upload EDI files to compare and forecast the load for the next vessel.")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_files = st.file_uploader("Upload EDI (.edi or .txt)", type=["edi", "txt"], accept_multiple_files=True)
    selected_files = []
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        selected_files = st.multiselect("Select Files (min 2):", file_names, default=file_names)
        num_clusters = st.slider("Clusters", 2, 10, 4)

if not uploaded_files or len(selected_files) < 2:
    st.info("ℹ️ Upload minimal 2 file untuk analisis.")
else:
    file_map = {f.name: f for f in uploaded_files}
    pivots_dict = {f_name: parse_edi_to_pivot(file_map[f_name]) for f_name in selected_files}

    with st.spinner("Processing..."):
        comparison_df = compare_multiple_pivots(pivots_dict)

    if comparison_df.empty:
        st.error("Tidak ada data valid.")
    else:
        st.header("📊 Forecast Summary Table")
        summary_table = create_summary_table(comparison_df)
        display_aggrid_centered(summary_table)

        st.header("🔀 Cluster Forecast (Boxes)")
        df_with_clusters = add_cluster_info(comparison_df, num_clusters)
        cluster_table = create_summarized_cluster_table(df_with_clusters)
        display_aggrid_centered(cluster_table)

        st.header("📦 Macro Slot Needs")
        macro_table = create_macro_slot_table(df_with_clusters)
        display_aggrid_centered(macro_table)

        st.markdown("---")
        st.subheader("📥 Export All Data")
        generate_excel_download_link({
            "Forecast Cluster": cluster_table,
            "Macro Slot": macro_table,
            "Detail Forecast": comparison_df
        })

        st.markdown("---")
        with st.expander("📋 Show Detail Forecast Table"):
            display_aggrid_centered(comparison_df)
