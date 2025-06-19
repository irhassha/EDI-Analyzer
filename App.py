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


def create_summary_table(pivots_dict, detailed_forecast_df):
    """
    Creates a summary table. The forecast column is now summed up from the
    detailed comparison table for consistency.
    """
    summaries = []
    for file_name, pivot in pivots_dict.items():
        if not pivot.empty:
            # Aggregate count and weight
            summary = pivot.groupby("Port of Discharge").agg(
                **{f'Count ({file_name})': ('Container Count', 'sum'), f'Weight ({file_name})': ('Total Weight', 'sum')}
            )
            summaries.append(summary)
            
    if not summaries:
        return pd.DataFrame()

    # Merge all historical summaries
    final_summary = pd.concat(summaries, axis=1).fillna(0).astype(int)
    
    # Calculate forecast per POD by summing from the detailed comparison
    if not detailed_forecast_df.empty:
        if 'Forecast (Next Vessel)' in detailed_forecast_df.columns:
            forecast_summary_count = detailed_forecast_df.groupby('Port of Discharge')['Forecast (Next Vessel)'].sum()
            final_summary['Forecast (Next Vessel)'] = forecast_summary_count
            final_summary['Forecast (Next Vessel)'] = final_summary['Forecast (Next Vessel)'].fillna(0)
        
        if 'Forecast Weight (KGM)' in detailed_forecast_df.columns:
            forecast_summary_weight = detailed_forecast_df.groupby('Port of Discharge')['Forecast Weight (KGM)'].sum()
            final_summary['Forecast Weight (KGM)'] = forecast_summary_weight
            final_summary['Forecast Weight (KGM)'] = final_summary['Forecast Weight (KGM)'].fillna(0)

    # Add a Total row
    total_row = final_summary.sum().to_frame().T
    total_row.index = ["**TOTAL**"]
    final_summary = pd.concat([final_summary, total_row]).astype(int)
    
    # Hide historical weight columns from the display
    cols_to_display = [col for col in final_summary.columns if not col.startswith('Weight')]
    
    return final_summary[cols_to_display].reset_index()

def add_cluster_info(df, num_clusters=6):
    """
    Adds cluster information to a DataFrame based on the 'Bay' column.
    """
    if df.empty or 'Bay' not in df.columns:
        return df

    df_clustered = df.copy()
    try:
        df_clustered['Cluster ID'] = pd.qcut(df_clustered['Bay'], q=num_clusters, labels=False, duplicates='drop')
    except ValueError:
        try:
            df_clustered['Cluster ID'], _ = pd.cut(df_clustered['Bay'], bins=num_clusters, labels=False, duplicates='drop')
        except ValueError:
            df_clustered['Cluster ID'] = 0 # Fallback if all else fails

    df_clustered['Bay Range'] = df_clustered.groupby('Cluster ID')['Bay'].transform(lambda x: f"{x.min()}-{x.max()}")
    return df_clustered

def create_summarized_cluster_table(comparison_df):
    """
    Creates a cluster summary table separating 20ft & 40ft containers,
    displaying the forecast count of boxes.
    """
    if comparison_df.empty or 'Forecast (Next Vessel)' not in comparison_df.columns:
        return pd.DataFrame()

    df = comparison_df.copy()
    df = df[df['Forecast (Next Vessel)'] > 0] 

    if df.empty:
        return pd.DataFrame()

    # Determine container type based on odd/even Bay (Odd=20, Even=40)
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    
    # Pivot to get the sum of forecast per cluster and allocation
    cluster_pivot = df.pivot_table(
        index=['Cluster ID', 'Bay Range'],
        columns='Allocation Column',
        values='Forecast (Next Vessel)', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Final formatting for the output table
    cluster_pivot = cluster_pivot.reset_index()
    cluster_pivot.drop(columns='Cluster ID', inplace=True)
    cluster_pivot.insert(0, 'CLUSTER', range(1, len(cluster_pivot) + 1))
    cluster_pivot = cluster_pivot.rename(columns={'Bay Range': 'BAY'})
    
    return cluster_pivot

def create_macro_slot_table(comparison_df):
    """
    Creates the Macro Slot Needs table based on the forecast.
    """
    if comparison_df.empty or 'Forecast (Next Vessel)' not in comparison_df.columns:
        return pd.DataFrame()

    df = comparison_df.copy()
    df = df[df['Forecast (Next Vessel)'] > 0]

    if df.empty:
        return pd.DataFrame()

    # Determine container type based on odd/even Bay
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    
    # Pivot to get the sum of forecast per cluster and allocation
    cluster_pivot = df.pivot_table(
        index=['Cluster ID', 'Bay Range'],
        columns='Allocation Column',
        values='Forecast (Next Vessel)',
        aggfunc='sum',
        fill_value=0
    )

    # --- SLOT CALCULATION LOGIC ---
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
    slot_df.drop(columns='Cluster ID', inplace=True)
    slot_df.insert(0, 'CLUSTER', range(1, len(slot_df) + 1))
    slot_df = slot_df.rename(columns={'Bay Range': 'BAY'})

    return slot_df

def create_colored_weight_chart(comparison_df):
    """
    Creates and displays a colored bar chart for the total forecast weight per Bay,
    colored by cluster.
    """
    if comparison_df.empty or 'Forecast Weight (KGM)' not in comparison_df.columns or 'Bay Range' not in comparison_df.columns:
        st.info("Not enough data to create the weight chart by cluster.")
        return

    weight_summary = comparison_df.groupby(['Bay', 'Bay Range'])['Forecast Weight (KGM)'].sum().reset_index()
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

    st.header(f"🔍 Detailed Comparison & Forecast")
    
    file_names = list(pivots_dict.keys())
    
    selected_files = st.multiselect(
        "Select files (in order) for comparison and forecast:",
        options=file_names,
        default=file_names
    )

    if len(selected_files) >= 2:
        pivots_to_compare = {name: pivots_dict[name] for name in selected_files}
        
        comparison_df = compare_multiple_pivots(pivots_to_compare)
        
        st.header("📊 Summary of Total Containers & Weight per Port of Discharge")
        summary_table = create_summary_table(pivots_dict, comparison_df)
        
        if not summary_table.empty:
            st.dataframe(summary_table.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
        else:
            st.warning("No valid data to create a summary.")
            
        st.markdown("---")

        title = " vs ".join([f"`{name}`" for name in selected_files])
        st.subheader(f"Detailed Comparison & Forecast Result: {title}")
        
        if not comparison_df.empty:
            # Add cluster information to the main DataFrame
            df_with_clusters = add_cluster_info(comparison_df)
            
            # Prepare the table for display (without weight columns)
            display_cols = [col for col in df_with_clusters.columns if not col.startswith('Weight') and 'Weight' not in col]
            st.dataframe(df_with_clusters[display_cols].style.set_properties(**{'text-align': 'center'}))

            st.markdown("---")
            st.header("🎯 Forecast Allocation Summary per Cluster (in Boxes)")
            cluster_table = create_summarized_cluster_table(df_with_clusters)
            
            if not cluster_table.empty:
                st.dataframe(cluster_table.style.set_properties(**{'text-align': 'center'}), use_container_width=True)

                st.markdown("---")
                st.header("⚙️ Macro Slot Needs")
                macro_slot_table = create_macro_slot_table(df_with_clusters)
                if not macro_slot_table.empty:
                    st.dataframe(macro_slot_table.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
            else:
                st.info("No forecast data to create the cluster allocation table.")
            
            st.markdown("---")
            st.header("⚖️ Forecast Weight (VGM) Chart per Bay")
            create_colored_weight_chart(df_with_clusters)

        else:
            st.warning(f"Cannot compare the selected files. Ensure the files are valid and contain data from POL 'IDJKT'.")
    else:
        st.warning("Please select at least 2 files to display the comparison.")
