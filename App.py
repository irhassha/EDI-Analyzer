import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt

# Konfigurasi halaman
st.set_page_config(page_title="EDI Forecaster", layout="wide")

# --- Fungsi Inti ---

def parse_edi_to_flat_df(uploaded_file):
    """
    Fungsi ini mengambil file EDI yang diunggah, mem-parsingnya,
    dan mengembalikannya sebagai DataFrame datar dengan satu baris per kontainer.
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
    df_display['Bay'] = pd.to_numeric(df_display['Bay'], errors='coerce')
    df_display.dropna(subset=['Bay'], inplace=True)
    df_display['Bay'] = df_display['Bay'].astype(int)
    
    return df_display

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

def create_summary_chart(comparison_df):
    """ Membuat grafik batang bertumpuk yang menampilkan jumlah total kontainer per sumber. """
    if comparison_df.empty: return
    summary_cols = [col for col in comparison_df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    grouping_cols = ['Port of Discharge'] + summary_cols
    summary_df = comparison_df[grouping_cols].groupby('Port of Discharge').sum().reset_index()
    try:
        melted_df = pd.melt(summary_df, id_vars=['Port of Discharge'], var_name='Source', value_name='Container Count')
    except KeyError:
        st.warning("Tidak dapat membuat grafik ringkasan karena data tidak ada.")
        return
    melted_df['Source Label'] = melted_df['Source'].str.replace('Count \(', '', regex=True).str.replace('\)', '', regex=True)
    totals_df = melted_df.groupby('Source Label')['Container Count'].sum().reset_index(name='Total Count')
    bars = alt.Chart(melted_df).mark_bar().encode(x=alt.X('Source Label:N', sort=None, title='Data Source (Vessel/Forecast)', axis=alt.Axis(labelAngle=0, labelLimit=200)), y=alt.Y('sum(Container Count):Q', title='Total Container Count'), color=alt.Color('Port of Discharge:N', title='Port of Discharge'), tooltip=['Source Label', 'Port of Discharge', 'Container Count'])
    text = alt.Chart(totals_df).mark_text(align='center', baseline='bottom', dy=-10, color='white').encode(x=alt.X('Source Label:N', sort=None), y=alt.Y('Total Count:Q'), text=alt.Text('Total Count:Q', format=','))
    chart = (bars + text).properties(title='Container Composition per Vessel and Forecast')
    st.altair_chart(chart, use_container_width=True)

def create_colored_weight_chart(df_with_clusters):
    """ Membuat grafik batang berwarna untuk total prediksi berat per Bay. """
    if df_with_clusters.empty or 'Forecast Weight (KGM)' not in df_with_clusters.columns or 'Bay Range' not in df_with_clusters.columns: return
    weight_summary = df_with_clusters.groupby(['Bay', 'Bay Range'])['Forecast Weight (KGM)'].sum().reset_index()
    weight_summary = weight_summary[weight_summary['Forecast Weight (KGM)'] > 0]
    if not weight_summary.empty:
        chart = alt.Chart(weight_summary).mark_bar().encode(x=alt.X('Bay:O', sort=None, title='Bay'), y=alt.Y('Forecast Weight (KGM):Q', title='Forecast Weight (KGM)'), color=alt.Color('Bay Range:N', title='Cluster'), tooltip=['Bay', 'Forecast Weight (KGM)', 'Bay Range']).properties(title='Forecast Weight (VGM) per Bay by Cluster')
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Tidak ada data prediksi berat untuk ditampilkan di grafik.")

def get_weight_class(weight, ranges):
    """ Menetapkan kelas berat berdasarkan rentang yang telah ditentukan. """
    if weight <= ranges['WC1']: return 'WC1'
    if weight <= ranges['WC2']: return 'WC2'
    if weight <= ranges['WC3']: return 'WC3'
    if weight <= ranges['WC4']: return 'WC4'
    return 'Overweight'

def create_wc_forecast_df(flat_dfs_dict, wc_ranges):
    """ Membuat DataFrame prediksi detail yang mencakup kelas berat. """
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

# --- TATA LETAK APLIKASI STREAMLIT ---
st.title("🚢 EDI File Comparator & Forecaster")
st.caption("Unggah file EDI untuk membandingkan, memprediksi, dan memvalidasi muatan untuk kapal berikutnya.")

with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")
    uploaded_files = st.file_uploader("1. Unggah file EDI historis", type=["edi", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        file_names = list(p.name for p in uploaded_files)
        selected_files = st.multiselect("2. Pilih file (secara berurutan):", options=file_names, default=file_names)
        
        all_pods = []
        if selected_files:
            try:
                pivots_for_pods = {f.name: parse_edi_to_flat_df(f) for f in uploaded_files if f.name in selected_files}
                all_pods = sorted(list(pd.concat([df['Port of Discharge'] for df in pivots_for_pods.values() if not df.empty]).unique()))
            except Exception as e:
                st.error(f"Error saat mengambil POD: {e}")
        
        excluded_pods = st.multiselect("3. Keluarkan Port of Discharge (opsional):", options=all_pods)
        
        with st.expander("Pengaturan Kelas Berat"):
            wc1 = st.number_input("Batas Atas WC1 (KGM)", value=9900)
            wc2 = st.number_input("Batas Atas WC2 (KGM)", value=15900)
            wc3 = st.number_input("Batas Atas WC3 (KGM)", value=21900)
            wc4 = st.number_input("Batas Atas WC4 (KGM)", value=30400)
            wc_ranges = {'WC1': wc1, 'WC2': wc2, 'WC3': wc3, 'WC4': wc4}
        
        num_clusters = st.number_input("5. Pilih jumlah kluster:", min_value=2, max_value=20, value=6, step=1, help="Ini akan mengelompokkan Bay ke dalam jumlah rentang yang dipilih untuk analisis.")


if not uploaded_files or len(uploaded_files) < 2:
    st.info("ℹ️ Silakan unggah minimal 2 file historis di sidebar untuk memulai analisis.")
elif 'selected_files' in locals() and len(selected_files) < 2:
    st.warning("Silakan pilih minimal 2 file historis di sidebar untuk menampilkan perbandingan.")
else:
    with st.spinner("Menganalisis file..."):
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
        st.error("Tidak dapat membuat perbandingan yang valid dari file yang dipilih. Silakan periksa file atau pengaturan Anda.")
    else:
        st.header("📊 Ringkasan per Kapal")
        create_summary_chart(comparison_df)
        st.markdown("---")

        st.header("🎯 Analisis Kluster & Kelas Berat")
        wc_forecast_df = create_wc_forecast_df(flat_dfs_dict, wc_ranges)
        df_with_clusters_and_wc = add_cluster_info(wc_forecast_df, num_clusters)
        
        sorted_clusters = df_with_clusters_and_wc.drop_duplicates(subset=['Cluster ID', 'Bay Range']).sort_values(by='Cluster ID')
        
        # Buat kolom untuk tata letak kartu
        num_total_clusters = len(sorted_clusters)
        if num_total_clusters > 0:
            cols = st.columns(min(num_total_clusters, 3))
            col_idx = 0
            for _, cluster_info in sorted_clusters.iterrows():
                cluster_id = cluster_info['Cluster ID']
                bay_range = cluster_info['Bay Range']
                
                cluster_data = df_with_clusters_and_wc[df_with_clusters_and_wc['Cluster ID'] == cluster_id]
                
                if not cluster_data.empty:
                    with cols[col_idx % 3]:
                        with st.container(border=True):
                            st.markdown(f"**Kluster: Bay {bay_range}**")

                            alloc_df = cluster_data.copy()
                            alloc_df['Container Type'] = np.where(alloc_df['Bay'] % 2 != 0, '20', '40')
                            
                            slot_df = alloc_df.copy()
                            slot_df['Slot Needs'] = np.ceil(slot_df['Forecast Count'] / 30)
                            slot_df.loc[slot_df['Container Type'] == '40', 'Slot Needs'] *= 2
                            
                            total_boxes = alloc_df['Forecast Count'].sum()
                            total_slots = slot_df['Slot Needs'].sum()
                            
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Total Prediksi Box", f"{total_boxes:,.0f}")
                            m_col2.metric("Total Kebutuhan Slot", f"{total_slots:,.0f}")
                            
                            st.markdown("---")
                            
                            # --- Pembuatan Tabel ---
                            for size in ['20', '40']:
                                size_data = alloc_df[alloc_df['Container Type'] == size]
                                if not size_data.empty and size_data['Forecast Count'].sum() > 0:
                                    st.markdown(f"**Kontainer {size}'**")
                                    
                                    pivot = size_data.pivot_table(
                                        index='Port of Discharge',
                                        columns='Weight Class',
                                        values='Forecast Count',
                                        aggfunc='sum',
                                        fill_value=0
                                    )
                                    # Hapus baris dan kolom yang semuanya nol
                                    pivot = pivot.loc[(pivot != 0).any(axis=1), (pivot != 0).any(axis=0)]
                                    
                                    if not pivot.empty:
                                        st.dataframe(pivot, use_container_width=True)
                col_idx += 1
        
        st.markdown("---")
        
        with st.expander("Tampilkan Tabel Perbandingan & Prediksi Detail"):
            display_cols = [col for col in comparison_df.columns if not col.startswith('Weight')]
            st.dataframe(comparison_df[display_cols], use_container_width=True)
            
        st.header("⚖️ Grafik Prediksi Berat (VGM) per Bay")
        df_with_clusters_for_chart = add_cluster_info(comparison_df, num_clusters)
        create_colored_weight_chart(df_with_clusters_for_chart)
