import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="EDI Forecaster", layout="wide")

# --- AGGRID UTILITY ---
def display_aggrid_centered(df, fit_columns=True):
    if df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(cellStyle={'textAlign': 'center'}, wrapText=True, autoHeight=True)
    grid_options = gb.build()
    AgGrid(df, gridOptions=grid_options, fit_columns_on_grid_load=fit_columns, theme="streamlit")

# --- ALL FUNCTIONS ---

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
        st.error("Could not find 'Port of Loading' information in the EDI file.")
        return pd.DataFrame()
    df_filtered = df_all.loc[df_all["Port of Loading"] == "IDJKT"].copy()
    if df_filtered.empty: return pd.DataFrame()
    df_display = df_filtered.drop(columns=["Port of Loading"]).dropna(subset=["Port of Discharge", "Bay"])
    df_display['Weight'] = df_display['Weight'].fillna(0)
    if df_display.empty: return pd.DataFrame()

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
    try:
        weighted_avg = np.average(y, weights=weights)
    except ZeroDivisionError:
        return round(np.mean(y))
    return max(0, round(weighted_avg))

def compare_multiple_pivots(pivots_dict_selected):
    if not pivots_dict_selected or len(pivots_dict_selected) < 2: return pd.DataFrame()
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

def create_summary_table(comparison_df):
    if comparison_df.empty: return pd.DataFrame()
    summary_cols = [col for col in comparison_df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    grouping_cols = ['Port of Discharge'] + summary_cols
    summary = comparison_df[grouping_cols].groupby('Port of Discharge').sum().reset_index()
    return summary

def add_cluster_info(df, num_clusters=6):
    if df.empty or 'Bay' not in df.columns or df['Bay'].nunique() < num_clusters:
        return df.assign(**{'Cluster ID': 0, 'Bay Range': 'N/A'})
    df_clustered = df.copy()
    try:
        df_clustered['Cluster ID'] = pd.qcut(df_clustered['Bay'], q=num_clusters, labels=False, duplicates='drop')
    except ValueError:
        df_clustered['Cluster ID'] = 0
    df_clustered['Bay Range'] = df_clustered.groupby('Cluster ID')['Bay'].transform(lambda x: f"{x.min()}-{x.max()}")
    return df_clustered

def create_summarized_cluster_table(df_with_clusters):
    if df_with_clusters.empty or 'Forecast (Next Vessel)' not in df_with_clusters.columns: return pd.DataFrame()
    df = df_with_clusters[df_with_clusters['Forecast (Next Vessel)'] > 0].copy()
    if df.empty: return pd.DataFrame()
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    cluster_pivot = df.pivot_table(index=['Bay Range'], columns='Allocation Column', values='Forecast (Next Vessel)', aggfunc='sum', fill_value=0)
    cluster_pivot['_sort_key'] = cluster_pivot.index.str.split('-').str[0].astype(int)
    cluster_pivot = cluster_pivot.sort_values(by='_sort_key').drop(columns='_sort_key')
    cluster_pivot['Total Boxes'] = cluster_pivot.sum(axis=1)
    cluster_pivot = cluster_pivot.reset_index().rename(columns={'Bay Range': 'BAY'})
    cluster_pivot.insert(0, 'CLUSTER', range(1, len(cluster_pivot) + 1))
    return cluster_pivot

def create_macro_slot_table(df_with_clusters):
    if df_with_clusters.empty or 'Forecast (Next Vessel)' not in df_with_clusters.columns: return pd.DataFrame()
    df = df_with_clusters[df_with_clusters['Forecast (Next Vessel)'] > 0].copy()
    if df.empty: return pd.DataFrame()
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    cluster_pivot = df.pivot_table(index=['Bay Range'], columns='Allocation Column', values='Forecast (Next Vessel)', aggfunc='sum', fill_value=0)
    slot_df = cluster_pivot.copy()
    for col in slot_df.columns:
        if ' 20' in col: slot_df[col] = np.ceil(slot_df[col] / 30)
        elif ' 40' in col: slot_df[col] = np.ceil(slot_df[col] / 30) * 2
    slot_df['Total Slot Needs'] = slot_df.sum(axis=1)
    slot_df = slot_df.astype(int)
    slot_df['_sort_key'] = slot_df.index.str.split('-').str[0].astype(int)
    slot_df = slot_df.sort_values(by='_sort_key').drop(columns='_sort_key')
    slot_df = slot_df.reset_index().drop(columns='Bay Range')
    slot_df.insert(0, 'CLUSTER', range(1, len(slot_df) + 1))
    total_col = slot_df.pop('Total Slot Needs')
    slot_df.insert(1, 'Total Slot Needs', total_col)
    return slot_df

def create_summary_chart(summary_df):
    if summary_df.empty: return
    try:
        melted_df = pd.melt(summary_df, id_vars=['Port of Discharge'], var_name='Source', value_name='Container Count')
    except KeyError:
        st.warning("Could not generate summary chart due to missing data.")
        return
    melted_df['Source Label'] = melted_df['Source'].str.replace('Count \(', '', regex=True).str.replace('\)', '', regex=True)
    totals_df = melted_df.groupby('Source Label')['Container Count'].sum().reset_index(name='Total Count')
    bars = alt.Chart(melted_df).mark_bar().encode(
        x=alt.X('Source Label:N', sort=None, title='Data Source (Vessel/Forecast)', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('sum(Container Count):Q', title='Total Container Count'),
        color=alt.Color('Port of Discharge:N', title='Port of Discharge'),
        tooltip=['Source Label', 'Port of Discharge', 'Container Count']
    )
    text = alt.Chart(totals_df).mark_text(align='center', baseline='bottom', dy=-10, color='white').encode(
        x=alt.X('Source Label:N', sort=None),
        y=alt.Y('Total Count:Q'),
        text=alt.Text('Total Count:Q', format=',')
    )
    st.altair_chart(bars + text, use_container_width=True)

def create_colored_weight_chart(df_with_clusters):
    if df_with_clusters.empty or 'Forecast Weight (KGM)' not in df_with_clusters.columns or 'Bay Range' not in df_with_clusters.columns:
        return
    weight_summary = df_with_clusters.groupby(['Bay', 'Bay Range'])['Forecast Weight (KGM)'].sum().reset_index()
    weight_summary = weight_summary[weight_summary['Forecast Weight (KGM)'] > 0]
    if not weight_summary.empty:
        chart = alt.Chart(weight_summary).mark_bar().encode(
            x=alt.X('Bay:O', sort=None, title='Bay'),
            y=alt.Y('Forecast Weight (KGM):Q', title='Forecast Weight (KGM)'),
            color=alt.Color('Bay Range:N', title='Cluster'),
            tooltip=['Bay', 'Forecast Weight (KGM)', 'Bay Range']
        ).properties(title='Forecast Weight (VGM) per Bay by Cluster')
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No forecast weight data to display in the chart.")

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

# --- APP LOGIC START ---
st.title("🚢 EDI File Comparator & Forecaster")
st.caption("Upload EDI files to compare and forecast the load for the next vessel.")

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    uploaded_files = st.file_uploader("1. Upload your .EDI files here", type=["edi", "txt"], accept_multiple_files=True)

    uploaded_file_buffers = {}
    if uploaded_files:
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
        num_clusters = st.number_input("4. Select number of clusters:", min_value=2, max_value=20, value=6)

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
        st.error("Could not generate a valid comparison from the selected files.")
    else:
        st.header("📊 Summary per Vessel")
        summary_table = create_summary_table(comparison_df)
        if not summary_table.empty:
            create_summary_chart(summary_table)

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

        with st.expander("Show Detailed Comparison & Forecast Table"):
            display_cols = [col for col in comparison_df.columns if not col.startswith('Weight')]
            display_aggrid_centered(comparison_df[display_cols])

        st.header("⚖️ Forecast Weight (VGM) Chart per Bay")
        create_colored_weight_chart(df_with_clusters)
