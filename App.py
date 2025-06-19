import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt

# Konfigurasi halaman
st.set_page_config(page_title="EDI Forecaster", layout="wide")

# --- Fungsi Inti ---

def parse_edi_to_pivot(uploaded_file):
    """
    Fungsi ini mengambil file EDI yang diunggah, mem-parsingnya,
    dan mengembalikannya sebagai DataFrame pivot.
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
        st.error("Tidak dapat menemukan informasi 'Port of Loading' di file EDI.")
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
    """ Memprediksi nilai berikutnya menggunakan Weighted Moving Average (WMA). """
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
    """ Membandingkan beberapa DataFrame pivot dan mengembalikan DataFrame gabungan dengan prediksi. """
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
    """ Membuat tabel ringkasan dari dataframe perbandingan utama. """
    if comparison_df.empty: return pd.DataFrame()
    summary_cols = [col for col in comparison_df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    grouping_cols = ['Port of Discharge'] + summary_cols
    summary = comparison_df[grouping_cols].groupby('Port of Discharge').sum().reset_index()
    return summary

def add_cluster_info(df, num_clusters=6):
    """ Menambahkan informasi kluster ke DataFrame. """
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
    """ Membuat tabel ringkasan kluster yang menampilkan jumlah kotak prediksi. """
    if df_with_clusters.empty or 'Forecast (Next Vessel)' not in df_with_clusters.columns: return pd.DataFrame()
    df = df_with_clusters[df_with_clusters['Forecast (Next Vessel)'] > 0].copy()
    if df.empty: return pd.DataFrame()
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    cluster_pivot = df.pivot_table(index=['Bay Range'], columns='Allocation Column', values='Forecast (Next Vessel)', aggfunc='sum', fill_value=0)
    
    # --- PERBAIKAN SORTING ---
    cluster_pivot['_sort_key'] = cluster_pivot.index.str.split('-').str[0].astype(int)
    cluster_pivot = cluster_pivot.sort_values(by='_sort_key').drop(columns='_sort_key')
    
    cluster_pivot = cluster_pivot.reset_index().rename(columns={'Bay Range': 'BAY'})
    cluster_pivot.insert(0, 'CLUSTER', range(1, len(cluster_pivot) + 1))
    return cluster_pivot

def create_macro_slot_table(df_with_clusters):
    """ Membuat tabel Macro Slot Needs berdasarkan prediksi. """
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
    
    # --- PERBAIKAN SORTING ---
    slot_df['_sort_key'] = slot_df.index.str.split('-').str[0].astype(int)
    slot_df = slot_df.sort_values(by='_sort_key').drop(columns='_sort_key')
    
    slot_df = slot_df.reset_index().rename(columns={'Bay Range': 'BAY'})
    slot_df.insert(0, 'CLUSTER', range(1, len(slot_df) + 1))
    return slot_df

def create_summary_chart(summary_df):
    """ Membuat grafik batang bertumpuk yang menampilkan jumlah total kontainer per sumber. """
    if summary_df.empty: return
    try:
        melted_df = pd.melt(summary_df, id_vars=['Port of Discharge'], var_name='Source', value_name='Container Count')
    except KeyError:
        st.warning("Tidak dapat membuat grafik ringkasan karena data tidak ada.")
        return

    melted_df['Source Label'] = melted_df['Source'].str.replace('Count \(', '', regex=True).str.replace('\)', '', regex=True)
    totals_df = melted_df.groupby('Source Label')['Container Count'].sum().reset_index(name='Total Count')

    bars = alt.Chart(melted_df).mark_bar().encode(
        x=alt.X('Source Label:N', sort=None, title='Data Source (Vessel/Forecast)', axis=alt.Axis(labelAngle=0, labelLimit=200)),
        y=alt.Y('sum(Container Count):Q', title='Total Container Count'),
        color=alt.Color('Port of Discharge:N', title='Port of Discharge'),
        tooltip=['Source Label', 'Port of Discharge', 'Container Count']
    )
    text = alt.Chart(totals_df).mark_text(align='center', baseline='bottom', dy=-10, color='white').encode(
        x=alt.X('Source Label:N', sort=None),
        y=alt.Y('Total Count:Q'),
        text=alt.Text('Total Count:Q', format=',')
    )
    chart = (bars + text).properties(title='Container Composition per Vessel and Forecast')
    st.altair_chart(chart, use_container_width=True)

def create_colored_weight_chart(df_with_clusters):
    """ Membuat grafik batang berwarna untuk total prediksi berat per Bay. """
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
        st.info("Tidak ada data prediksi berat untuk ditampilkan di grafik.")
        
def style_table(df, use_pod_colors=False):
    """ Menerapkan gaya rata tengah dan pewarnaan header (opsional) pada DataFrame. """
    styles = [{'selector': 'th, td', 'props': [('text-align', 'center')]}]
    
    if use_pod_colors:
        pods = set()
        for col in df.columns:
            if isinstance(col, str):
                parts = col.split()
                if len(parts) > 1:
                    pods.add(parts[0])

        colors = ['#2E4053', '#566573', '#34495E', '#212F3D', '#515A5A', '#85929E']
        color_map = {pod: colors[i % len(colors)] for i, pod in enumerate(sorted(list(pods)))}

        for i, col_name in enumerate(df.columns):
            pod_in_col = next((pod for pod in color_map if pod in col_name), None)
            if pod_in_col:
                styles.append({
                    'selector': f'th.col_heading.level0.col{i}',
                    'props': [('background-color', color_map[pod_in_col]), ('color', 'white')]
                })
    return df.style.set_table_styles(styles)

# --- TAMPILAN APLIKASI STREAMLIT ---

st.title("🚢 EDI File Comparator & Forecaster")
st.caption("Unggah file EDI untuk membandingkan dan memprediksi muatan kapal berikutnya.")

with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")
    uploaded_files = st.file_uploader("1. Unggah file .EDI Anda di sini", type=["edi", "txt"], accept_multiple_files=True)
    if uploaded_files:
        file_names = list(p.name for p in uploaded_files)
        selected_files = st.multiselect("2. Pilih file (secara berurutan):", options=file_names, default=file_names)
        
        all_pods = []
        if selected_files:
            try:
                pivots_for_pods = {f.name: parse_edi_to_pivot(f) for f in uploaded_files if f.name in selected_files}
                all_pods = sorted(list(pd.concat([df['Port of Discharge'] for df in pivots_for_pods.values() if not df.empty]).unique()))
            except Exception as e:
                st.error(f"Error saat mengambil POD: {e}")
        
        excluded_pods = st.multiselect("3. Keluarkan Port of Discharge (opsional):", options=all_pods)
        
        num_clusters = st.number_input("4. Pilih jumlah kluster:", min_value=2, max_value=20, value=6, step=1, help="Ini akan mengelompokkan Bay ke dalam jumlah rentang yang dipilih untuk analisis.")


if not uploaded_files or len(uploaded_files) < 2:
    st.info("ℹ️ Silakan unggah minimal 2 file di sidebar untuk memulai analisis.")
elif 'selected_files' in locals() and len(selected_files) < 2:
    st.warning("Silakan pilih minimal 2 file di sidebar untuk menampilkan perbandingan.")
else:
    with st.spinner("Menganalisis file..."):
        if 'pivots_for_pods' in locals() and all(f in pivots_for_pods for f in selected_files):
             pivots_dict = {name: pivots_for_pods[name] for name in selected_files}
        else:
             pivots_dict = {f.name: parse_edi_to_pivot(f) for f in uploaded_files if f.name in selected_files}

        comparison_df = compare_multiple_pivots(pivots_dict)
        
        if excluded_pods:
            comparison_df = comparison_df[~comparison_df['Port of Discharge'].isin(excluded_pods)]

    if comparison_df.empty:
        st.error("Tidak dapat membuat perbandingan yang valid dari file yang dipilih. Silakan periksa file atau pengaturan Anda.")
    else:
        st.header("📊 Ringkasan per Kapal")
        summary_table = create_summary_table(comparison_df)
        if not summary_table.empty:
            create_summary_chart(summary_table)
        else:
            st.warning("Tidak ada data valid untuk membuat ringkasan.")
            
        st.markdown("---")

        st.header("🎯 Analisis Kluster")
        df_with_clusters = add_cluster_info(comparison_df, num_clusters)
        
        st.subheader("Ringkasan Prediksi Alokasi per Kluster (dalam Box)")
        cluster_table = create_summarized_cluster_table(df_with_clusters)
        if not cluster_table.empty:
            st.dataframe(style_table(cluster_table.set_index('CLUSTER'), use_pod_colors=True), use_container_width=True)

        st.subheader("Kebutuhan Slot Makro")
        macro_slot_table = create_macro_slot_table(df_with_clusters)
        if not macro_slot_table.empty:
            st.dataframe(style_table(macro_slot_table.set_index('CLUSTER'), use_pod_colors=True), use_container_width=True)
        
        st.markdown("---")
        
        with st.expander("Tampilkan Tabel Perbandingan & Prediksi Detail"):
            display_cols = [col for col in comparison_df.columns if not col.startswith('Weight')]
            st.dataframe(style_table(comparison_df[display_cols]), use_container_width=True)
            
        st.header("⚖️ Grafik Prediksi Berat (VGM) per Bay")
        create_colored_weight_chart(df_with_clusters)
