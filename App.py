import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt

# Page configuration
st.set_page_config(page_title="EDI File Comparator", layout="wide")

# --- CORE FUNCTIONS ---

def parse_edi_to_pivot(uploaded_file):
    """
    This function takes an uploaded EDI file, parses it,
    and returns it as a pivot DataFrame.
    """
    try:
        # Move the file cursor back to the beginning each time this function is called
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8")
        lines = content.strip().splitlines()
    except Exception:
        # Return an empty DataFrame if the file cannot be read
        return pd.DataFrame()

    all_records = []
    current_container_data = {}

    for line in lines:
        line = line.strip()
        if line.startswith("LOC+147+"):
            if current_container_data:
                all_records.append(current_container_data)
            # Initialize data for the new container
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
                # Remove apostrophe and other non-numeric characters before conversion
                weight_str = line.split(':')[-1].replace("'", "").strip()
                current_container_data['Weight'] = pd.to_numeric(weight_str, errors='coerce')
            except (IndexError, ValueError):
                pass # Ignore if format is incorrect


    if current_container_data:
        all_records.append(current_container_data)

    if not all_records:
        return pd.DataFrame()

    df_all = pd.DataFrame(all_records)
    # Filter for Port of Loading 'IDJKT' only
    if "Port of Loading" not in df_all.columns:
        st.error("Could not find 'Port of Loading' information in the EDI file.")
        return pd.DataFrame()
        
    df_filtered = df_all.loc[df_all["Port of Loading"] == "IDJKT"].copy()

    if df_filtered.empty:
        return pd.DataFrame()
        
    df_display = df_filtered.drop(columns=["Port of Loading"]).dropna(subset=["Port of Discharge", "Bay"])
    df_display['Weight'] = df_display['Weight'].fillna(0)
    
    if df_display.empty:
        return pd.DataFrame()

    # Aggregate count and weight
    pivot_df = df_display.groupby(["Bay", "Port of Discharge"], as_index=False).agg(
        **{'Container Count': ('Bay', 'size'), 'Total Weight': ('Weight', 'sum')}
    )
    
    # Ensure Bay column is numeric for sorting and clustering
    pivot_df['Bay'] = pd.to_numeric(pivot_df['Bay'], errors='coerce')
    pivot_df.dropna(subset=['Bay'], inplace=True)
    pivot_df['Bay'] = pivot_df['Bay'].astype(int)
    
    return pivot_df


def forecast_next_value_wma(series):
    """
    Forecasts the next value in a series using a Weighted Moving Average (WMA).
    More recent data is given a higher weight.
    """
    # Requires at least 2 data points for a meaningful forecast
    if len(series.dropna()) < 2:
        # Return a simple average if less than 2 data points
        return round(series.mean()) if not series.empty else 0

    y = series.values
    
    # Create linearly increasing weights (1, 2, 3, ..., n)
    weights = np.arange(1, len(y) + 1)
    
    try:
        # Calculate the weighted average
        weighted_avg = np.average(y, weights=weights)
    except ZeroDivisionError:
        return round(np.mean(y)) # Fallback if total weight is zero

    # Forecast cannot be negative and should be an integer
    return max(0, round(weighted_avg))


def compare_multiple_pivots(pivots_dict_selected):
    """
    Compares multiple pivot DataFrames and returns a combined
    comparison DataFrame with forecast columns.
    """
    if not pivots_dict_selected or len(pivots_dict_selected) < 2:
        return pd.DataFrame()

    # Prepare each DataFrame to be merged
    dfs_to_merge = []
    for name, df in pivots_dict_selected.items():
        if not df.empty:
            # Set Bay and POD as index and rename columns
            renamed_df = df.set_index(["Bay", "Port of Discharge"])
            renamed_df = renamed_df.rename(columns={
                "Container Count": f"Count ({name})",
                "Total Weight": f"Weight ({name})"
            })
            dfs_to_merge.append(renamed_df)

    if not dfs_to_merge:
        return pd.DataFrame()

    # Merge all DataFrames with an outer join
    merged_df = pd.concat(dfs_to_merge, axis=1, join='outer')
    merged_df = merged_df.fillna(0).astype(int)

    # Calculate forecast columns for Count and Weight
    count_cols = [col for col in merged_df.columns if col.startswith('Count')]
    weight_cols = [col for col in merged_df.columns if col.startswith('Weight')]
    
    if count_cols:
        merged_df['Forecast (Next Vessel)'] = merged_df[count_cols].apply(forecast_next_value_wma, axis=1).astype(int)
    if weight_cols:
        merged_df['Forecast Weight (KGM)'] = merged_df[weight_cols].apply(forecast_next_value_wma, axis=1).astype(int)

    # Sort by Bay and return a flat table
    merged_df = merged_df.reset_index()
    merged_df = merged_df.sort_values(by="Bay").reset_index(drop=True)

    return merged_df


def create_summary_table(comparison_df):
    """
    Creates a summary table from the main comparison dataframe.
    """
    if comparison_df.empty:
        return pd.DataFrame()

    # Identify all relevant columns (count and forecast)
    summary_cols = [col for col in comparison_df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    # Add Port of Discharge for grouping
    grouping_cols = ['Port of Discharge'] + summary_cols
    
    # Group by POD and sum up the values
    summary = comparison_df[grouping_cols].groupby('Port of Discharge').sum().reset_index()

    # Add a Total row
    total_row = summary[summary_cols].sum().to_frame().T
    total_row['Port of Discharge'] = "**TOTAL**"
    
    final_summary = pd.concat([summary, total_row], ignore_index=True)
    
    return final_summary


def add_cluster_info(df, num_clusters=6):
    """
    Adds cluster information to a DataFrame based on the 'Bay' column.
    """
    if df.empty or 'Bay' not in df.columns or df['Bay'].nunique() < num_clusters:
        return df.assign(**{'Cluster ID': 0, 'Bay Range': 'N/A'})

    df_clustered = df.copy()
    try:
        df_clustered['Cluster ID'] = pd.qcut(df_clustered['Bay'], q=num_clusters, labels=False, duplicates='drop')
    except ValueError:
        try:
            df_clustered['Cluster ID'], _ = pd.cut(df_clustered['Bay'], bins=num_clusters, labels=False, duplicates='drop')
        except ValueError:
            # Fallback if clustering fails completely
            df_clustered['Cluster ID'] = 0 

    df_clustered['Bay Range'] = df_clustered.groupby('Cluster ID')['Bay'].transform(lambda x: f"{x.min()}-{x.max()}")
    return df_clustered

def create_summarized_cluster_table(df_with_clusters):
    """
    Creates a cluster summary table separating 20ft & 40ft containers,
    displaying the forecast count of boxes.
    """
    if df_with_clusters.empty or 'Forecast (Next Vessel)' not in df_with_clusters.columns:
        return pd.DataFrame()

    df = df_with_clusters.copy()
    df = df[df['Forecast (Next Vessel)'] > 0] 

    if df.empty:
        return pd.DataFrame()

    # Determine container type based on odd/even Bay (Odd=20, Even=40)
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    
    # Pivot to get the sum of forecast per cluster and allocation
    cluster_pivot = df.pivot_table(
        index=['Bay Range'],
        columns='Allocation Column',
        values='Forecast (Next Vessel)', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Final formatting for the output table
    cluster_pivot = cluster_pivot.reset_index()
    cluster_pivot.insert(0, 'CLUSTER', range(1, len(cluster_pivot) + 1))
    cluster_pivot = cluster_pivot.rename(columns={'Bay Range': 'BAY'})
    
    return cluster_pivot

def create_macro_slot_table(df_with_clusters):
    """
    Creates the Macro Slot Needs table based on the forecast.
    """
    if df_with_clusters.empty or 'Forecast (Next Vessel)' not in df_with_clusters.columns:
        return pd.DataFrame()

    df = df_with_clusters.copy()
    df = df[df['Forecast (Next Vessel)'] > 0]

    if df.empty:
        return pd.DataFrame()

    # Determine container type based on odd/even Bay
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    
    # Pivot to get the sum of forecast per cluster and allocation
    cluster_pivot = df.pivot_table(
        index=['Bay Range'],
        columns='Allocation Column',
        values='Forecast (Next Vessel)',
        aggfunc='sum',
        fill_value=0
    )

    # SLOT CALCULATION LOGIC
    slot_df = cluster_pivot.copy()
    for col in slot_df.columns:
        if ' 20' in col:
            # Slot need for 20ft is boxes/30, rounded up
            slot_df[col] = np.ceil(slot_df[col] / 30)
        elif ' 40' in col:
            # Slot need for 40ft is (boxes/30, rounded up) * 2
            slot_df[col] = np.ceil(slot_df[col] / 30) * 2
    
    # Calculate total slot needs per cluster
    slot_df['Total Slot Needs'] = slot_df.sum(axis=1)

    # Final formatting
    slot_df = slot_df.astype(int).reset_index()
    slot_df.insert(0, 'CLUSTER', range(1, len(slot_df) + 1))
    slot_df = slot_df.rename(columns={'Bay Range': 'BAY'})

    return slot_df

def create_colored_weight_chart(df_with_clusters):
    """
    Creates and displays a colored bar chart for the total forecast weight per Bay,
    colored by cluster.
    """
    if df_with_clusters.empty or 'Forecast Weight (KGM)' not in df_with_clusters.columns or 'Bay Range' not in df_with_clusters.columns:
        st.info("Not enough data to create the weight chart by cluster.")
        return

    weight_summary = df_with_clusters.groupby(['Bay', 'Bay Range'])['Forecast Weight (KGM)'].sum().reset_index()
    weight_summary = weight_summary[weight_summary['Forecast Weight (KGM)'] > 0]

    if not weight_summary.empty:
        chart = alt.Chart(weight_summary).mark_bar().encode(
            x=alt.X('Bay:O', sort=None, title='Bay'),
            y=alt.Y('Forecast Weight (KGM):Q', title='Forecast Weight (KGM)'),
            color=alt.Color('Bay Range:N', title='Cluster'),
            tooltip=['Bay', 'Forecast Weight (KGM)', 'Bay Range']
        ).properties(
            title='Forecast Weight (VGM) per Bay by Cluster'
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No forecast weight data to display in the chart.")

def create_summary_chart(summary_df):
    """
    Creates and displays a stacked bar chart for the summary data, showing composition per vessel.
    """
    # Exclude the TOTAL row for charting
    chart_data = summary_df[summary_df['Port of Discharge'] != '**TOTAL**'].copy()

    if chart_data.empty:
        return

    # Melt the dataframe to have a long format suitable for Altair
    try:
        melted_df = pd.melt(
            chart_data,
            id_vars=['Port of Discharge'],
            var_name='Source',
            value_name='Container Count'
        )
    except KeyError:
        # This can happen if there are no count/forecast columns
        st.warning("Could not generate summary chart due to missing data.")
        return


    # Clean up the 'Source' names for the legend
    melted_df['Source'] = melted_df['Source'].str.replace('Count \(', '', regex=True).str.replace('\)', '', regex=True)

    chart = alt.Chart(melted_df).mark_bar().encode(
        x=alt.X('Source:N', sort=None, title='Data Source (Vessel/Forecast)'),
        y=alt.Y('Container Count:Q', title='Total Container Count'),
        color=alt.Color('Port of Discharge:N', title='Port of Discharge'),
        tooltip=['Source', 'Port of Discharge', 'Container Count']
    ).properties(
        title='Container Composition per Vessel and Forecast'
    )
    st.altair_chart(chart, use_container_width=True)

def style_dataframe_left(df):
    """
    Applies left alignment to both headers and cells of a DataFrame.
    """
    return df.style.set_properties(**{'text-align': 'left'}).set_table_styles([
        {'selector': 'th, td', 'props': [('text-align', 'left')]}
    ])

# --- STREAMLIT APP LAYOUT ---

st.title("🚢 EDI File Comparator & Forecaster")
st.caption("Upload EDI files to compare and forecast the load for the next vessel.")

uploaded_files = st.file_uploader(
    "Upload your .EDI files here",
    type=["edi", "txt"],
    accept_multiple_files=True
)

if len(uploaded_files) < 2:
    st.info("ℹ️ Please upload at least 2 files to start the comparison and forecast.")
else:
    with st.spinner("Analyzing files..."):
        pivots_dict = {f.name: parse_edi_to_pivot(f) for f in uploaded_files}

    st.header(f"⚙️ Analysis Settings")
    
    file_names = list(pivots_dict.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_files = st.multiselect(
            "1. Select files (in order) for comparison:",
            options=file_names,
            default=file_names
        )
    
    # Get all unique PODs from the selected files
    all_pods = sorted(list(pd.concat([pivots_dict[name]['Port of Discharge'] for name in selected_files if name in pivots_dict and not pivots_dict[name].empty]).unique()))
    
    with col2:
        excluded_pods = st.multiselect(
            "2. Exclude Ports of Discharge (optional):",
            options=all_pods
        )

    if len(selected_files) >= 2:
        pivots_to_compare = {name: pivots_dict[name] for name in selected_files}
        
        comparison_df = compare_multiple_pivots(pivots_to_compare)
        
        # Filter out excluded PODs
        if excluded_pods:
            comparison_df = comparison_df[~comparison_df['Port of Discharge'].isin(excluded_pods)]

        st.markdown("---")
        st.header("📊 Summary of Total Containers per Port of Discharge")
        summary_table = create_summary_table(comparison_df)
        
        if not summary_table.empty:
            create_summary_chart(summary_table)
        else:
            st.warning("No valid data to create a summary.")
            
        st.markdown("---")

        title = " vs ".join([f"`{name}`" for name in selected_files])
        st.subheader(f"Detailed Comparison & Forecast Result: {title}")
        
        if not comparison_df.empty:
            # Prepare the table for display (without weight columns)
            display_cols = [col for col in comparison_df.columns if not col.startswith('Weight')]
            st.dataframe(style_dataframe_left(comparison_df[display_cols]), use_container_width=True)

            st.markdown("---")
            st.header("🎯 Cluster Analysis")
            
            num_clusters = st.number_input(
                "Select number of clusters:",
                min_value=2, max_value=20, value=6, step=1,
                help="This will group the Bays into the selected number of ranges for analysis."
            )
            
            # Add cluster information based on user input
            df_with_clusters = add_cluster_info(comparison_df, num_clusters)

            st.subheader("Forecast Allocation Summary per Cluster (in Boxes)")
            cluster_table = create_summarized_cluster_table(df_with_clusters)
            
            if not cluster_table.empty:
                st.dataframe(style_dataframe_left(cluster_table.set_index('CLUSTER')), use_container_width=True)

                st.subheader("Macro Slot Needs")
                macro_slot_table = create_macro_slot_table(df_with_clusters)
                if not macro_slot_table.empty:
                    st.dataframe(style_dataframe_left(macro_slot_table.set_index('CLUSTER')), use_container_width=True)
            else:
                st.info("No forecast data to create the cluster allocation table.")
            
            st.markdown("---")
            st.header("⚖️ Forecast Weight (VGM) Chart per Bay")
            create_colored_weight_chart(df_with_clusters)

        else:
            st.warning(f"Cannot compare the selected files. Ensure the files are valid and contain data from POL 'IDJKT'.")
    else:
        st.warning("Please select at least 2 files to display the comparison.")
