import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt

# Page configuration
st.set_page_config(page_title="EDI Forecaster", layout="wide")

# --- CORE FUNCTIONS ---

def parse_edi_to_flat_df(uploaded_file):
    """
    This function takes an uploaded EDI file, parses it,
    and returns a flat DataFrame with one row per container.
    """
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
    df_display['Bay'] = pd.to_numeric(df_display['Bay'], errors='coerce')
    df_display.dropna(subset=['Bay'], inplace=True)
    df_display['Bay'] = df_display['Bay'].astype(int)
    
    return df_display

def forecast_next_value_wma(series):
    """ Forecasts the next value using a Weighted Moving Average (WMA). """
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
    """ Compares multiple pivot DataFrames and returns a combined DataFrame with forecasts. """
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
    """ Creates a summary table from the main comparison dataframe. """
    if comparison_df.empty: return pd.DataFrame()
    summary_cols = [col for col in comparison_df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    grouping_cols = ['Port of Discharge'] + summary_cols
    summary = comparison_df[grouping_cols].groupby('Port of Discharge').sum().reset_index()
    return summary

def add_cluster_info(df, num_clusters=6):
    """ Adds cluster information to a DataFrame. """
    if df.empty or 'Bay' not in df.columns or df['Bay'].nunique() < num_clusters:
        return df.assign(**{'Cluster ID': 0, 'Bay Range': 'N/A'})
    df_clustered = df.copy()
    try:
        df_clustered['Cluster ID'] = pd.qcut(df_clustered['Bay'], q=num_clusters, labels=False, duplicates='drop')
    except ValueError:
        df_clustered['Cluster ID'] = 0
    df_clustered['Bay Range'] = df_clustered.groupby('Cluster ID')['Bay'].transform(lambda x: f"{x.min()}-{x.max()}")
    return df_clustered

def create_summary_chart(comparison_df):
    """ Creates a stacked bar chart showing total container counts per source. """
    if comparison_df.empty: return
    summary_cols = [col for col in comparison_df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    grouping_cols = ['Port of Discharge'] + summary_cols
    summary_df = comparison_df[grouping_cols].groupby('Port of Discharge').sum().reset_index()
    try:
        melted_df = pd.melt(summary_df, id_vars=['Port of Discharge'], var_name='Source', value_name='Container Count')
    except KeyError:
        st.warning("Could not generate summary chart due to missing data.")
        return
    melted_df['Source Label'] = melted_df['Source'].str.replace('Count \(', '', regex=True).str.replace('\)', '', regex=True)
    totals_df = melted_df.groupby('Source Label')['Container Count'].sum().reset_index(name='Total Count')
    bars = alt.Chart(melted_df).mark_bar().encode(x=alt.X('Source Label:N', sort=None, title='Data Source (Vessel/Forecast)', axis=alt.Axis(labelAngle=0, labelLimit=200)), y=alt.Y('sum(Container Count):Q', title='Total Container Count'), color=alt.Color('Port of Discharge:N', title='Port of Discharge'), tooltip=['Source Label', 'Port of Discharge', 'Container Count'])
    text = alt.Chart(totals_df).mark_text(align='center', baseline='bottom', dy=-10, color='white').encode(x=alt.X('Source Label:N', sort=None), y=alt.Y('Total Count:Q'), text=alt.Text('Total Count:Q', format=','))
    chart = (bars + text).properties(title='Container Composition per Vessel and Forecast')
    st.altair_chart(chart, use_container_width=True)

def create_colored_weight_chart(df_with_clusters):
    """ Creates a colored bar chart for the total forecast weight per Bay. """
    if df_with_clusters.empty or 'Forecast Weight (KGM)' not in df_with_clusters.columns or 'Bay Range' not in df_with_clusters.columns: return
    weight_summary = df_with_clusters.groupby(['Bay', 'Bay Range'])['Forecast Weight (KGM)'].sum().reset_index()
    weight_summary = weight_summary[weight_summary['Forecast Weight (KGM)'] > 0]
    if not weight_summary.empty:
        chart = alt.Chart(weight_summary).mark_bar().encode(x=alt.X('Bay:O', sort=None, title='Bay'), y=alt.Y('Forecast Weight (KGM):Q', title='Forecast Weight (KGM)'), color=alt.Color('Bay Range:N', title='Cluster'), tooltip=['Bay', 'Forecast Weight (KGM)', 'Bay Range']).properties(title='Forecast Weight (VGM) per Bay by Cluster')
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No forecast weight data to display in the chart.")

def get_weight_class(weight, ranges):
    """ Assigns a weight class based on predefined ranges. """
    if weight <= ranges['WC1']: return 'WC1'
    if weight <= ranges['WC2']: return 'WC2'
    if weight <= ranges['WC3']: return 'WC3'
    if weight <= ranges['WC4']: return 'WC4'
    return 'Overweight'

def create_wc_forecast_df(flat_dfs_dict, wc_ranges):
    """ Creates a detailed forecast DataFrame that includes weight classes. """
    dfs_to_merge = []
    for name, df in flat_dfs_dict.items():
        if not df.empty:
            df['Weight Class'] = df['Weight'].apply(lambda w: get_weight_class(w, wc_ranges))
            summary = df.groupby(['Bay', 'Port of Discharge', 'Weight Class']).size().reset_index(name=f"Count ({name})")
            dfs_to_merge.append(summary.set_index(['Bay', 'Port of Discharge', 'Weight Class']))

    if not dfs_to_merge: return pd.DataFrame()

    merged_df = pd.concat(dfs_to_merge, axis=1, join='outer').fillna(0).astype(int)
    count_cols = [col for col in merged_df.columns if col.startswith('Count')]
    if count_cols:
        merged_df['Forecast Count'] = merged_df[count_cols].apply(forecast_next_value_wma, axis=1).astype(int)
    
    return merged_df.reset_index()

# --- STREAMLIT APP LAYOUT ---
st.title("🚢 EDI File Comparator & Forecaster")
st.caption("Upload EDI files to compare, forecast, and validate the load for the next vessel.")

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    uploaded_files = st.file_uploader("1. Upload historical EDI files", type=["edi", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        file_names = list(p.name for p in uploaded_files)
        selected_files = st.multiselect("2. Select files (in order):", options=file_names, default=file_names)
        
        all_pods = []
        if selected_files:
            try:
                pivots_for_pods = {f.name: parse_edi_to_flat_df(f) for f in uploaded_files if f.name in selected_files}
                all_pods = sorted(list(pd.concat([df['Port of Discharge'] for df in pivots_for_pods.values() if not df.empty]).unique()))
            except Exception as e:
                st.error(f"Error getting PODs: {e}")
        
        excluded_pods = st.multiselect("3. Exclude Ports of Discharge (optional):", options=all_pods)
        
        with st.expander("Weight Class Settings"):
            wc1 = st.number_input("WC1 Upper Limit (KGM)", value=9900)
            wc2 = st.number_input("WC2 Upper Limit (KGM)", value=15900)
            wc3 = st.number_input("WC3 Upper Limit (KGM)", value=21900)
            wc4 = st.number_input("WC4 Upper Limit (KGM)", value=30400)
            wc_ranges = {'WC1': wc1, 'WC2': wc2, 'WC3': wc3, 'WC4': wc4}
        
        num_clusters = st.number_input("5. Select number of clusters:", min_value=2, max_value=20, value=6, step=1, help="This will group the Bays into the selected number of ranges for analysis.")


if not uploaded_files or len(uploaded_files) < 2:
    st.info("ℹ️ Please upload at least 2 historical files in the sidebar to start the analysis.")
elif 'selected_files' in locals() and len(selected_files) < 2:
    st.warning("Please select at least 2 historical files in the sidebar to display the comparison.")
else:
    with st.spinner("Analyzing files..."):
        if 'pivots_for_pods' in locals() and all(f in pivots_for_pods for f in selected_files):
             flat_dfs_dict = {name: pivots_for_pods[name] for name in selected_files}
        else:
             flat_dfs_dict = {f.name: parse_edi_to_flat_df(f) for f in uploaded_files if f.name in selected_files}

        pivots_dict = {}
        for name, df in flat_dfs_dict.items():
            pivots_dict[name] = df.groupby(["Bay", "Port of Discharge"], as_index=False).agg(
                **{'Container Count': ('Bay', 'size'), 'Total Weight': ('Weight', 'sum')}
            )
        comparison_df = compare_multiple_pivots(pivots_dict)
        
        if excluded_pods:
            comparison_df = comparison_df[~comparison_df['Port of Discharge'].isin(excluded_pods)]
            for name in flat_dfs_dict:
                flat_dfs_dict[name] = flat_dfs_dict[name][~flat_dfs_dict[name]['Port of Discharge'].isin(excluded_pods)]

    if comparison_df.empty:
        st.error("Could not generate a valid comparison from the selected files. Please check the files or your settings.")
    else:
        st.header("📊 Summary per Vessel")
        create_summary_chart(comparison_df)
        st.markdown("---")

        st.header("🎯 Cluster & Weight Class Analysis")
        df_with_clusters = add_cluster_info(comparison_df, num_clusters)
        
        wc_forecast_df = create_wc_forecast_df(flat_dfs_dict, wc_ranges)
        df_with_clusters_and_wc = add_cluster_info(wc_forecast_df, num_clusters)
        
        sorted_clusters = df_with_clusters.groupby('Cluster ID')['Bay'].min().sort_values().index
        
        for cluster_id in sorted_clusters:
            cluster_data = df_with_clusters_and_wc[df_with_clusters_and_wc['Cluster ID'] == cluster_id]
            if not cluster_data.empty:
                bay_range = cluster_data['Bay Range'].iloc[0]
                with st.container():
                    st.subheader(f"Cluster: Bay {bay_range}")
                    
                    # Allocation Table
                    alloc_df = cluster_data.copy()
                    alloc_df['Container Type'] = np.where(alloc_df['Bay'] % 2 != 0, '20', '40')
                    alloc_pivot = alloc_df.pivot_table(index='Port of Discharge', columns=['Container Type', 'Weight Class'], values='Forecast Count', aggfunc='sum', fill_value=0)
                    if not alloc_pivot.empty:
                        st.write("**Forecast Allocation (in Boxes)**")
                        st.dataframe(alloc_pivot, use_container_width=True)

                    # Slot Needs Table
                    slot_df = alloc_pivot.copy()
                    for col in slot_df.columns:
                        if '20' in col[0]: slot_df[col] = np.ceil(slot_df[col] / 30)
                        elif '40' in col[0]: slot_df[col] = np.ceil(slot_df[col] / 30) * 2
                    slot_df['Total Slot Needs'] = slot_df.sum(axis=1)
                    if not slot_df.empty:
                        st.write("**Macro Slot Needs**")
                        st.dataframe(slot_df.astype(int), use_container_width=True)
        
        st.markdown("---")
        
        with st.expander("Show Detailed Comparison & Forecast Table"):
            display_cols = [col for col in comparison_df.columns if not col.startswith('Weight')]
            st.dataframe(comparison_df[display_cols], use_container_width=True)
            
        st.header("⚖️ Forecast Weight (VGM) Chart per Bay")
        create_colored_weight_chart(df_with_clusters)
