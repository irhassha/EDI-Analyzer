import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt
import re
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(page_title="EDI Forecaster", layout="wide")

# --- Fungsi Inti ---

def parse_edi_to_flat_df(uploaded_file, file_type='export'):
    """
    Fungsi ini mengambil file EDI yang diunggah, mem-parsingnya,
    dan mengembalikannya sebagai DataFrame datar dengan satu baris per kontainer.
    Mendukung tipe file 'export' (loading) dan 'import' (discharge).
    """
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8", errors='ignore')
        # --- PERBAIKAN: Memisahkan data berdasarkan apostrof, bukan baris baru ---
        lines = content.strip().split("'")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return pd.DataFrame()

    all_records = []
    current_container_data = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("LOC+147+"):
            if current_container_data.get("Bay"): # Hanya simpan jika ada data bay
                all_records.append(current_container_data)
            current_container_data = {'Weight': 0}
            try:
                full_bay = line.split("+")[2].split(":")[0]
                current_container_data["Bay"] = full_bay[1:3] if len(full_bay) >= 3 else full_bay
            except IndexError:
                current_container_data["Bay"] = None
        elif line.startswith("LOC+11+"):
            try:
                pod = line.split("+")[2].split(":")[0].strip()
                current_container_data["Port of Discharge"] = pod
            except IndexError:
                pass
        elif line.startswith("LOC+9+"):
            try:
                pol = line.split("+")[2].split(":")[0].strip()
                current_container_data["Port of Loading"] = pol
            except IndexError:
                pass
        elif line.startswith("MEA+VGM++KGM:"):
            try:
                weight_str = line.split(':')[-1].strip()
                current_container_data['Weight'] = pd.to_numeric(weight_str, errors='coerce')
            except (IndexError, ValueError):
                pass
        elif line.startswith("EQD+CN+"):
             try:
                size_code = line.split('+')[3]
                if size_code.startswith('2'):
                    current_container_data['Size'] = '20'
                elif size_code.startswith('4'):
                    current_container_data['Size'] = '40'
                else:
                    current_container_data['Size'] = 'Other'
             except IndexError:
                current_container_data['Size'] = 'Unknown'

    if current_container_data.get("Bay"):
        all_records.append(current_container_data)

    if not all_records:
        return pd.DataFrame()

    df_all = pd.DataFrame(all_records)

    # Terapkan filter berdasarkan tipe file
    if file_type == 'export':
        if "Port of Loading" not in df_all.columns:
            return pd.DataFrame()
        df_filtered = df_all[df_all["Port of Loading"].str.strip().str.upper() == "IDJKT"].copy()
    elif file_type == 'import':
        if "Port of Discharge" not in df_all.columns:
            return pd.DataFrame()
        df_filtered = df_all[df_all["Port of Discharge"].str.strip().str.upper() == "IDJKT"].copy()
    else:
        return pd.DataFrame()

    if df_filtered.empty: return pd.DataFrame()
    
    # Menghapus baris dengan nilai "Bay" yang hilang dan mengubah tipe data
    df_filtered.dropna(subset=["Bay"], inplace=True)
    df_filtered['Weight'] = df_filtered['Weight'].fillna(0)
    df_filtered['Bay'] = pd.to_numeric(df_filtered['Bay'], errors='coerce')
    df_filtered.dropna(subset=['Bay'], inplace=True)
    df_filtered['Bay'] = df_filtered['Bay'].astype(int)
    
    return df_filtered

def forecast_next_value_wma(series):
    """ Memperkirakan nilai berikutnya menggunakan Weighted Moving Average (WMA). """
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
    """ Membandingkan beberapa Pivot DataFrame dan mengembalikan DataFrame gabungan dengan perkiraan. """
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
    """ Membuat tabel ringkasan dari DataFrame perbandingan utama. """
    if comparison_df.empty: return pd.DataFrame()
    summary_cols = [col for col in comparison_df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    grouping_cols = ['Port of Discharge'] + summary_cols
    summary = comparison_df[grouping_cols].groupby('Port of Discharge').sum().reset_index()
    return summary

def add_cluster_info(df, num_clusters=6):
    """ Menambahkan informasi cluster ke DataFrame. """
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
    """ Membuat diagram batang bertumpuk yang menunjukkan jumlah kontainer per sumber. """
    if comparison_df.empty: return
    summary_cols = [col for col in comparison_df.columns if col.startswith('Count') or 'Forecast (Next Vessel)' in col]
    grouping_cols = ['Port of Discharge'] + summary_cols
    summary_df = comparison_df[grouping_cols].groupby('Port of Discharge').sum().reset_index()
    try:
        melted_df = pd.melt(summary_df, id_vars=['Port of Discharge'], var_name='Source', value_name='Container Count')
    except KeyError:
        st.warning("Tidak dapat membuat diagram ringkasan karena tidak ada data.")
        return

    # Ekstrak nama file asli dari 'Source'
    melted_df['Original_File_Name'] = melted_df['Source'].apply(
        lambda x: re.search(r'Count \((.*?)\)', x).group(1) if 'Count (' in x else x
    )

    # Label Sumber yang Disederhanakan tanpa tanggal
    melted_df['Source Label'] = melted_df['Original_File_Name']

    totals_df = melted_df.groupby('Source Label')['Container Count'].sum().reset_index(name='Total Count')
    bars = alt.Chart(melted_df).mark_bar().encode(
        x=alt.X('Source Label:N', sort=None, title='Sumber Data (Kapal/Perkiraan)', axis=alt.Axis(labelAngle=-45, labelLimit=200)),
        y=alt.Y('sum(Container Count):Q', title='Jumlah Total Kontainer'),
        color=alt.Color('Port of Discharge:N', title='Port of Discharge'),
        tooltip=['Source Label', 'Port of Discharge', 'Container Count']
    )
    text = alt.Chart(totals_df).mark_text(align='center', baseline='bottom', dy=-10, color='white').encode(
        x=alt.X('Source Label:N', sort=None),
        y=alt.Y('Total Count:Q'),
        text=alt.Text('Total Count:Q', format=',')
    )
    chart = (bars + text).properties(title='Komposisi Kontainer per Kapal dan Perkiraan')
    st.altair_chart(chart, use_container_width=True)

def create_colored_weight_chart(df_with_clusters):
    """ Membuat diagram batang berwarna untuk total perkiraan berat per Bay. """
    if df_with_clusters.empty or 'Forecast Weight (KGM)' not in df_with_clusters.columns or 'Bay Range' not in df_with_clusters.columns: return
    weight_summary = df_with_clusters.groupby(['Bay', 'Bay Range'])['Forecast Weight (KGM)'].sum().reset_index()
    weight_summary = weight_summary[weight_summary['Forecast Weight (KGM)'] > 0]
    if not weight_summary.empty:
        chart = alt.Chart(weight_summary).mark_bar().encode(x=alt.X('Bay:O', sort=None, title='Bay'), y=alt.Y('Forecast Weight (KGM):Q', title='Forecast Weight (KGM)'), color=alt.Color('Bay Range:N', title='Cluster'), tooltip=['Bay', 'Forecast Weight (KGM)', 'Bay Range']).properties(title='Forecast Weight (VGM) per Bay berdasarkan Cluster')
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Tidak ada data perkiraan berat untuk ditampilkan dalam diagram.")

def get_weight_class(weight, ranges):
    """ Menetapkan kelas berat berdasarkan rentang yang ditentukan. """
    if weight <= ranges['WC1']: return 'WC1'
    if weight <= ranges['WC2']: return 'WC2'
    if weight <= ranges['WC3']: return 'WC3'
    if weight <= ranges['WC4']: return 'WC4'
    return 'Overweight'

def create_wc_forecast_df(flat_dfs_dict, wc_ranges):
    """ Membuat DataFrame perkiraan detail yang mencakup kelas berat. """
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

def extract_date_from_filename(filename):
    """
    Mengekstrak tanggal dari nama file dengan format YYYYMMDD.
    Contoh: EDI_EXPORT_20250730.edi -> 2025-07-30
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d').date()
        except ValueError:
            pass
    return None

def extract_date_from_edi_content(edi_content):
    """
    Mengekstrak tanggal dari konten file EDI dengan mencari baris DTM+137:YYMMDD.
    """
    match = re.search(r"DTM\+137:(\d{6})", edi_content)
    if match:
        try:
            return datetime.strptime(match.group(1), '%y%m%d').date()
        except ValueError:
            pass
    return None

# --- STREAMLIT APP LAYOUT ---
st.title("🚢 EDI File Analyzer")
st.caption("Analisis file EDI untuk perencanaan loading (export) dan discharge (import).")

tab1, tab2 = st.tabs(["Export Forecast", "Import Analysis"])

with tab1:
    with st.sidebar:
        st.header("⚙️ Pengaturan Analisis Ekspor")
        uploaded_files = st.file_uploader("1. Unggah file EDI historis (Ekspor)", type=["edi", "txt"], accept_multiple_files=True)
        
        selected_files = []
        if uploaded_files:
            file_dates = {}
            for f in uploaded_files:
                file_date = extract_date_from_filename(f.name)
                
                # Jika parsing nama file gagal, coba parsing dari konten file
                if not file_date:
                    try:
                        f.seek(0)
                        content = f.read().decode("utf-8", errors='ignore')
                        file_date = extract_date_from_edi_content(content)
                    except Exception as e:
                        st.warning(f"Error saat membaca konten file '{f.name}': {e}")
                
                if file_date:
                    file_dates[f.name] = file_date
                else:
                    file_dates[f.name] = datetime.today().date()
                    st.warning(f"Tidak dapat mengekstrak tanggal dari nama file atau konten file untuk '{f.name}'. Menggunakan tanggal hari ini sebagai cadangan.")
            
            # Tampilkan tanggal yang diekstrak untuk debugging
            with st.expander("Tanggal File yang Diekstrak"):
                st.write(file_dates)

            sorted_file_names = sorted(file_dates, key=file_dates.get)
            selected_files = st.multiselect("2. Pilih file (urutkan secara berurutan berdasarkan tanggal):", options=sorted_file_names, default=sorted_file_names)
            
            all_pods = []
            if selected_files:
                try:
                    selected_uploaded_files = [f for f in uploaded_files if f.name in selected_files]
                    pivots_for_pods = {f.name: parse_edi_to_flat_df(f, file_type='export') for f in selected_uploaded_files}
                    all_pods = sorted(list(pd.concat([df['Port of Discharge'] for df in pivots_for_pods.values() if not df.empty]).unique()))
                except Exception as e:
                    st.error(f"Error saat mengambil POD: {e}")
            
            excluded_pods = st.multiselect("3. Kecualikan Port of Discharge (opsional):", options=all_pods)
            
            with st.expander("Pengaturan Kelas Berat"):
                wc1 = st.number_input("Batas Atas WC1 (KGM)", value=9900)
                wc2 = st.number_input("Batas Atas WC2 (KGM)", value=15900)
                wc3 = st.number_input("Batas Atas WC3 (KGM)", value=21900)
                wc4 = st.number_input("Batas Atas WC4 (KGM)", value=30400)
                wc_ranges = {'WC1': wc1, 'WC2': wc2, 'WC3': wc3, 'WC4': wc4}
            
            num_clusters = st.number_input("4. Pilih jumlah cluster:", min_value=2, max_value=20, value=6, step=1, help="Ini akan mengelompokkan Bay ke dalam rentang yang dipilih untuk analisis.")

    if not uploaded_files or len(uploaded_files) < 2:
        st.info("ℹ️ Silakan unggah setidaknya 2 file historis di sidebar untuk memulai analisis ekspor.")
    elif len(selected_files) < 2:
        st.warning("Silakan pilih setidaknya 2 file historis di sidebar untuk menampilkan perbandingan.")
    else:
        with st.spinner("Menganalisis file ekspor..."):
            try:
                selected_uploaded_files_for_parsing = [f for name in selected_files for f in uploaded_files if f.name == name]
                flat_dfs_dict = {f.name: parse_edi_to_flat_df(f, file_type='export') for f in selected_uploaded_files_for_parsing}
                
                # Filter out any empty dataframes before processing
                flat_dfs_dict = {name: df for name, df in flat_dfs_dict.items() if not df.empty}
                
                if len(flat_dfs_dict) < 2:
                    st.warning("Setelah memfilter, tidak ada cukup file dengan data yang valid untuk perbandingan.")
                else:
                    pivots_dict = {}
                    for name, df in flat_dfs_dict.items():
                        pivots_dict[name] = df.groupby(["Bay", "Port of Discharge"], as_index=False).agg(
                            **{'Container Count': ('Bay', 'size'), 'Total Weight': ('Weight', 'sum')}
                        )
                    comparison_df = compare_multiple_pivots(pivots_dict)
                    
                    if excluded_pods:
                        comparison_df = comparison_df[~comparison_df['Port of Discharge'].isin(excluded_pods)]
                    
                    if comparison_df.empty:
                        st.error("Tidak dapat membuat perbandingan yang valid dari file yang dipilih. Silakan periksa file atau pengaturan Anda.")
                    else:
                        st.header("📊 Ringkasan per Kapal")
                        create_summary_chart(comparison_df)
                        st.markdown("---")

                        st.header("🎯 Analisis Cluster & Kelas Berat")
                        wc_forecast_df = create_wc_forecast_df(flat_dfs_dict, wc_ranges)
                        df_with_clusters_and_wc = add_cluster_info(wc_forecast_df, num_clusters)
                        
                        sorted_clusters = df_with_clusters_and_wc.drop_duplicates(subset=['Cluster ID', 'Bay Range']).sort_values(by='Cluster ID')
                        
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
                                            st.markdown(f"**Cluster: Bay {bay_range}**")
                                            alloc_df = cluster_data.copy()
                                            alloc_df['Container Type'] = np.where(alloc_df['Bay'] % 2 != 0, '20', '40')
                                            total_boxes = alloc_df['Forecast Count'].sum()
                                            st.metric("Total Kotak Perkiraan", f"{total_boxes:,.0f}")
                                            st.markdown("---")
                                            for size in ['20', '40']:
                                                size_data = alloc_df[alloc_df['Container Type'] == size]
                                                if not size_data.empty and size_data['Forecast Count'].sum() > 0:
                                                    st.markdown(f"**Kontainer {size}'**")
                                                    pivot = size_data.groupby('Port of Discharge')['Forecast Count'].sum().reset_index()
                                                    pivot.rename(columns={'Forecast Count': 'Box Count'}, inplace=True)
                                                    st.dataframe(pivot, use_container_width=True, hide_index=True)
                                col_idx += 1
                        
                        st.markdown("---")
                        
                        with st.expander("Tampilkan Tabel Perbandingan & Perkiraan Detail"):
                            display_cols = [col for col in comparison_df.columns if not col.startswith('Weight')]
                            st.dataframe(comparison_df[display_cols], use_container_width=True)
                            
                        st.header("⚖️ Grafik Perkiraan Berat (VGM) per Bay")
                        df_with_clusters_for_chart = add_cluster_info(comparison_df, num_clusters)
                        create_colored_weight_chart(df_with_clusters_for_chart)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}. Silakan periksa file yang Anda unggah.")


with tab2:
    st.header("📥 Analisis Bongkar (Discharge)")
    
    discharge_file = st.file_uploader("Unggah file EDI Discharge di sini", type=["edi", "txt"])
    
    if discharge_file:
        with st.spinner("Menganalisis file discharge..."):
            import_df = parse_edi_to_flat_df(discharge_file, file_type='import')
        
        if not import_df.empty:
            st.subheader(f"Ringkasan Bongkar untuk IDJKT dari file: `{discharge_file.name}`")
            
            total_units = len(import_df)
            count_20ft = len(import_df[import_df['Size'] == '20'])
            count_40ft = len(import_df[import_df['Size'] == '40'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Unit Bongkar", f"{total_units}")
            col2.metric("Jumlah 20ft", f"{count_20ft}")
            col3.metric("Jumlah 40ft", f"{count_40ft}")
            
        else:
            st.warning("Tidak ada data bongkar untuk IDJKT yang ditemukan di file ini.")
    else:
        st.info("Silakan unggah file EDI discharge untuk melihat ringkasan bongkar.")
