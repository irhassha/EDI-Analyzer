import streamlit as st
import pandas as pd
import numpy as np
import io

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="EDI File Comparator", layout="wide")

# --- FUNGSI UTAMA ---

def parse_edi_to_pivot(uploaded_file):
    """
    Fungsi ini mengambil file EDI yang diunggah, mem-parsingnya,
    dan mengembalikannya sebagai DataFrame pivot.
    """
    try:
        # Pindahkan kursor file kembali ke awal setiap kali fungsi ini dipanggil
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8")
        lines = content.strip().splitlines()
    except Exception:
        # Jika file tidak bisa dibaca, kembalikan DataFrame kosong
        return pd.DataFrame()

    all_records = []
    current_container_data = {}

    for line in lines:
        line = line.strip()
        if line.startswith("LOC+147+"):
            if current_container_data:
                all_records.append(current_container_data)
            # Inisialisasi data untuk kontainer baru
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
        # --- PERUBAHAN: Membaca data berat dari VGM ---
        elif line.startswith("MEA+VGM++KGM:"):
            try:
                # Menghapus apostrof dan karakter non-numerik lainnya sebelum konversi
                weight_str = line.split(':')[-1].replace("'", "").strip()
                current_container_data['Weight'] = pd.to_numeric(weight_str, errors='coerce')
            except (IndexError, ValueError):
                pass # Abaikan jika format salah


    if current_container_data:
        all_records.append(current_container_data)

    if not all_records:
        return pd.DataFrame()

    df_all = pd.DataFrame(all_records)
    # Filter untuk Port of Loading 'IDJKT' saja
    df_filtered = df_all.loc[df_all["Port of Loading"] == "IDJKT"].copy()


    if df_filtered.empty:
        return pd.DataFrame()
        
    df_display = df_filtered.drop(columns=["Port of Loading"]).dropna(subset=["Port of Discharge", "Bay"])
    df_display['Weight'] = df_display['Weight'].fillna(0)
    
    if df_display.empty:
        return pd.DataFrame()

    # --- Agregasi jumlah dan berat ---
    pivot_df = df_display.groupby(["Bay", "Port of Discharge"], as_index=False).agg(
        **{'Jumlah Kontainer': ('Bay', 'size'), 'Total Berat': ('Weight', 'sum')}
    )
    
    # Pastikan kolom Bay adalah numerik untuk sorting dan clustering
    pivot_df['Bay'] = pd.to_numeric(pivot_df['Bay'], errors='coerce')
    pivot_df.dropna(subset=['Bay'], inplace=True)
    pivot_df['Bay'] = pivot_df['Bay'].astype(int)
    
    return pivot_df


def forecast_next_value_wma(series):
    """
    Memperkirakan nilai berikutnya menggunakan Weighted Moving Average (WMA).
    Data yang lebih baru akan memiliki bobot yang lebih besar.
    """
    # Membutuhkan minimal 2 titik data untuk membuat prediksi yang berarti
    if len(series.dropna()) < 2:
        # Jika kurang dari 2 data, kembalikan rata-rata sederhana
        return round(series.mean()) if not series.empty else 0

    y = series.values
    
    # Membuat bobot yang meningkat secara linear (1, 2, 3, ..., n)
    weights = np.arange(1, len(y) + 1)
    
    try:
        # Menghitung rata-rata tertimbang
        weighted_avg = np.average(y, weights=weights)
    except ZeroDivisionError:
        return round(np.mean(y)) # Fallback jika total bobot adalah nol

    # Forecast tidak boleh negatif dan harus integer
    return max(0, round(weighted_avg))


def compare_multiple_pivots(pivots_dict_selected):
    """
    Fungsi ini membandingkan beberapa DataFrame pivot dan mengembalikan
    DataFrame perbandingan dengan kolom forecast.
    """
    if not pivots_dict_selected or len(pivots_dict_selected) < 2:
        return pd.DataFrame()

    # Mempersiapkan setiap DataFrame untuk digabungkan
    dfs_to_merge = []
    for name, df in pivots_dict_selected.items():
        if not df.empty:
            # Jadikan Bay dan POD sebagai index dan ganti nama kolom
            renamed_df = df.set_index(["Bay", "Port of Discharge"])
            renamed_df = renamed_df.rename(columns={
                "Jumlah Kontainer": f"Jumlah ({name})",
                "Total Berat": f"Berat ({name})"
            })
            dfs_to_merge.append(renamed_df)

    if not dfs_to_merge:
        return pd.DataFrame()

    # Menggabungkan semua DataFrame dengan outer join
    merged_df = pd.concat(dfs_to_merge, axis=1, join='outer')
    merged_df = merged_df.fillna(0).astype(int)

    # Menghitung kolom forecast untuk Jumlah dan Berat
    jumlah_cols = [col for col in merged_df.columns if col.startswith('Jumlah')]
    berat_cols = [col for col in merged_df.columns if col.startswith('Berat')]
    
    if jumlah_cols:
        merged_df['Forecast (Next Vessel)'] = merged_df[jumlah_cols].apply(forecast_next_value_wma, axis=1).astype(int)
    if berat_cols:
        merged_df['Forecast Weight (KGM)'] = merged_df[berat_cols].apply(forecast_next_value_wma, axis=1).astype(int)


    # Mengurutkan berdasarkan Bay dan mengembalikan ke bentuk tabel datar
    merged_df = merged_df.reset_index()
    merged_df = merged_df.sort_values(by="Bay").reset_index(drop=True)

    return merged_df


def create_summary_table(pivots_dict, detailed_forecast_df):
    """
    Membuat tabel ringkasan. Kolom forecast sekarang dijumlahkan dari
    tabel perbandingan detail untuk konsistensi.
    """
    summaries = []
    for file_name, pivot in pivots_dict.items():
        if not pivot.empty:
            # Agregasi jumlah dan berat
            summary = pivot.groupby("Port of Discharge").agg(
                **{f'Jumlah ({file_name})': ('Jumlah Kontainer', 'sum'), f'Berat ({file_name})': ('Total Berat', 'sum')}
            )
            summaries.append(summary)
            
    if not summaries:
        return pd.DataFrame()

    # Gabungkan semua summary historis
    final_summary = pd.concat(summaries, axis=1).fillna(0).astype(int)
    
    # Hitung forecast per POD dengan menjumlahkan dari perbandingan detail
    if not detailed_forecast_df.empty:
        if 'Forecast (Next Vessel)' in detailed_forecast_df.columns:
            forecast_summary_count = detailed_forecast_df.groupby('Port of Discharge')['Forecast (Next Vessel)'].sum()
            final_summary['Forecast (Next Vessel)'] = forecast_summary_count
            final_summary['Forecast (Next Vessel)'] = final_summary['Forecast (Next Vessel)'].fillna(0)
        
        if 'Forecast Weight (KGM)' in detailed_forecast_df.columns:
            forecast_summary_weight = detailed_forecast_df.groupby('Port of Discharge')['Forecast Weight (KGM)'].sum()
            final_summary['Forecast Weight (KGM)'] = forecast_summary_weight
            final_summary['Forecast Weight (KGM)'] = final_summary['Forecast Weight (KGM)'].fillna(0)

    # Tambahkan baris Total
    total_row = final_summary.sum().to_frame().T
    total_row.index = ["**TOTAL**"]
    final_summary = pd.concat([final_summary, total_row]).astype(int)
    
    return final_summary.reset_index()

def create_summarized_cluster_table(comparison_df, num_clusters=6):
    """
    Membuat tabel ringkasan kluster yang memisahkan kontainer 20ft & 40ft
    dan menampilkannya sebagai jumlah kotak (forecast).
    """
    if comparison_df.empty or 'Forecast (Next Vessel)' not in comparison_df.columns:
        return pd.DataFrame()

    df = comparison_df.copy()
    df = df[df['Forecast (Next Vessel)'] > 0] # Hanya proses baris dengan forecast

    if df.empty:
        return pd.DataFrame()

    # Tentukan tipe kontainer berdasarkan Bay ganjil/genap (Ganjil=20, Genap=40)
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')

    # Buat bin/kelompok untuk cluster
    try:
        # Menggunakan qcut untuk distribusi yang lebih merata jika memungkinkan
        df['Cluster ID'] = pd.qcut(df['Bay'], q=num_clusters, labels=False, duplicates='drop')
    except ValueError:
        try:
             # Fallback ke cut jika qcut gagal
            df['Cluster ID'], bins = pd.cut(df['Bay'], bins=num_clusters, retbins=True, right=True, include_lowest=True, labels=False, duplicates='drop')
        except ValueError:
             return pd.DataFrame() # Gagal membuat kluster

    # Buat label rentang yang mudah dibaca dari setiap kluster
    df['BAY'] = df.groupby('Cluster ID')['Bay'].transform(lambda x: f"{x.min()}-{x.max()}")
    
    # Buat nama kolom tujuan (e.g., 'SGSIN 20')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    
    # Pivot untuk mendapatkan jumlah forecast per cluster dan alokasi
    cluster_pivot = df.pivot_table(
        index=['Cluster ID', 'BAY'],
        columns='Allocation Column',
        values='Forecast (Next Vessel)', # Langsung menggunakan jumlah forecast
        aggfunc='sum',
        fill_value=0
    )
    
    # Final formatting untuk tabel output
    cluster_pivot = cluster_pivot.reset_index()
    cluster_pivot.drop(columns='Cluster ID', inplace=True)
    cluster_pivot.insert(0, 'CLUSTER', range(1, len(cluster_pivot) + 1))

    return cluster_pivot

def create_macro_slot_table(comparison_df, num_clusters=6):
    """
    Membuat tabel kebutuhan slot makro berdasarkan forecast.
    """
    if comparison_df.empty or 'Forecast (Next Vessel)' not in comparison_df.columns:
        return pd.DataFrame()

    df = comparison_df.copy()
    df = df[df['Forecast (Next Vessel)'] > 0]

    if df.empty:
        return pd.DataFrame()

    # Tentukan tipe kontainer berdasarkan Bay ganjil/genap
    df['Container Type'] = np.where(df['Bay'] % 2 != 0, '20', '40')

    # Buat bin/kelompok untuk cluster
    try:
        df['Cluster ID'] = pd.qcut(df['Bay'], q=num_clusters, labels=False, duplicates='drop')
    except ValueError:
        try:
            df['Cluster ID'], bins = pd.cut(df['Bay'], bins=num_clusters, retbins=True, right=True, include_lowest=True, labels=False, duplicates='drop')
        except ValueError:
            return pd.DataFrame()

    # Buat label rentang yang mudah dibaca
    df['BAY'] = df.groupby('Cluster ID')['Bay'].transform(lambda x: f"{x.min()}-{x.max()}")
    
    # Buat nama kolom tujuan
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']
    
    # Pivot untuk mendapatkan jumlah forecast per cluster dan alokasi
    cluster_pivot = df.pivot_table(
        index=['Cluster ID', 'BAY'],
        columns='Allocation Column',
        values='Forecast (Next Vessel)',
        aggfunc='sum',
        fill_value=0
    )

    # --- INI BAGIAN BARU: KALKULASI SLOT ---
    slot_df = cluster_pivot.copy()
    for col in slot_df.columns:
        if ' 20' in col:
            # Slot need untuk 20ft adalah boxes/30, dibulatkan ke atas
            slot_df[col] = np.ceil(slot_df[col] / 30)
        elif ' 40' in col:
            # Slot need untuk 40ft adalah (boxes/30, dibulatkan ke atas) * 2
            slot_df[col] = np.ceil(slot_df[col] / 30) * 2
    
    # Hitung total slot needs per cluster
    slot_df['Total Slot Needs'] = slot_df.sum(axis=1)

    # Final formatting
    slot_df = slot_df.astype(int).reset_index()
    slot_df.drop(columns='Cluster ID', inplace=True)
    slot_df.insert(0, 'CLUSTER', range(1, len(slot_df) + 1))

    return slot_df

def create_weight_chart(comparison_df):
    """
    Membuat dan menampilkan bar chart untuk total prediksi berat per Bay.
    """
    if comparison_df.empty or 'Forecast Weight (KGM)' not in comparison_df.columns:
        st.info("Tidak ada data prediksi berat untuk ditampilkan di grafik.")
        return

    weight_summary = comparison_df.groupby('Bay')['Forecast Weight (KGM)'].sum()
    weight_summary = weight_summary[weight_summary > 0] # Hanya tampilkan bay dengan berat

    if not weight_summary.empty:
        st.bar_chart(weight_summary)
    else:
        st.info("Tidak ada data prediksi berat untuk ditampilkan di grafik.")


# --- TAMPILAN APLIKASI STREAMLIT ---

st.title("🚢 EDI File Comparator & Forecaster")
st.caption("Unggah file EDI untuk membandingkan dan memprediksi muatan kapal berikutnya.")

uploaded_files = st.file_uploader(
    "Upload file .EDI Anda di sini",
    type=["edi", "txt"],
    accept_multiple_files=True
)

if len(uploaded_files) < 2:
    st.info("ℹ️ Silakan unggah minimal 2 file untuk memulai perbandingan dan prediksi.")
else:
    # Memproses setiap file yang diunggah
    with st.spinner("Menganalisis file..."):
        pivots_dict = {f.name: parse_edi_to_pivot(f) for f in uploaded_files}

    # --- UI UNTUK MEMILIH FILE YANG AKAN DIBANDINGKAN ---
    st.header(f"🔍 Perbandingan Detail & Prediksi")
    
    file_names = list(pivots_dict.keys())
    
    selected_files = st.multiselect(
        "Pilih file (secara berurutan) untuk perbandingan dan prediksi:",
        options=file_names,
        default=file_names  # Defaultnya memilih semua file
    )

    # --- MENAMPILKAN HASIL PERBANDINGAN BERDASARKAN PILIHAN ---
    if len(selected_files) >= 2:
        pivots_to_compare = {name: pivots_dict[name] for name in selected_files}
        
        comparison_df = compare_multiple_pivots(pivots_to_compare)
        
        # --- MENAMPILKAN RINGKASAN SETELAH PERBANDINGAN SELESAI ---
        st.header("📊 Ringkasan Total Kontainer & Berat per Port of Discharge")
        # Melewatkan hasil perbandingan ke fungsi ringkasan untuk konsistensi
        summary_table = create_summary_table(pivots_dict, comparison_df)
        
        if not summary_table.empty:
            # Menerapkan perataan tengah pada tabel ringkasan
            st.dataframe(summary_table.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
        else:
            st.warning("Tidak ada data valid untuk membuat ringkasan.")
            
        st.markdown("---")

        # Membuat judul dinamis
        title = " vs ".join([f"`{name}`" for name in selected_files])
        st.subheader(f"Hasil Perbandingan Detail & Prediksi: {title}")
        
        if not comparison_df.empty:
            # Menerapkan perataan tengah pada tabel perbandingan
            st.dataframe(comparison_df.style.set_properties(**{'text-align': 'center'}))

            # --- BUAT DAN TAMPILKAN TABEL ALOKASI CLUSTER ---
            st.markdown("---")
            st.header("🎯 Ringkasan Prediksi Alokasi per Cluster (dalam Box)")
            cluster_table = create_summarized_cluster_table(comparison_df)
            
            if not cluster_table.empty:
                st.dataframe(cluster_table.style.set_properties(**{'text-align': 'center'}), use_container_width=True)

                # --- BUAT DAN TAMPILKAN TABEL MACRO SLOT NEEDS ---
                st.markdown("---")
                st.header("⚙️ Macro Slot Needs")
                macro_slot_table = create_macro_slot_table(comparison_df)
                if not macro_slot_table.empty:
                    st.dataframe(macro_slot_table.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
            else:
                st.info("Tidak ada data forecast untuk membuat tabel alokasi cluster.")
            
            # --- BUAT DAN TAMPILKAN GRAFIK BERAT ---
            st.markdown("---")
            st.header("⚖️ Grafik Prediksi Berat (VGM) per Bay")
            create_weight_chart(comparison_df)

        else:
            st.warning(f"Tidak dapat membandingkan file-file yang dipilih. Pastikan file valid dan berisi data dari POL 'IDJKT'.")
    else:
        st.warning("Silakan pilih minimal 2 file untuk ditampilkan perbandingannya.")
