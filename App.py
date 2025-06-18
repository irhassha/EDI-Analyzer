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
            current_container_data = {}
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
    
    if df_display.empty:
        return pd.DataFrame()

    # Membuat pivot table
    pivot_df = df_display.groupby(["Bay", "Port of Discharge"]).size().reset_index(name="Jumlah Kontainer")
    
    # Pastikan kolom Bay adalah numerik untuk sorting dan clustering
    pivot_df['Bay'] = pd.to_numeric(pivot_df['Bay'], errors='coerce')
    pivot_df.dropna(subset=['Bay'], inplace=True)
    pivot_df['Bay'] = pivot_df['Bay'].astype(int)
    
    return pivot_df


def forecast_next_value(series):
    """
    Memperkirakan nilai berikutnya dalam sebuah series menggunakan regresi linear.
    """
    # Membutuhkan minimal 2 titik data untuk membuat tren
    if len(series.dropna()) < 2:
        # Jika kurang dari 2 data, kembalikan rata-rata
        return round(series.mean()) if not series.empty else 0

    y = series.values
    x = np.arange(len(y))

    # Regresi linear untuk menemukan slope (m) dan intercept (b)
    try:
        m, b = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        # Jika terjadi error (misal, semua data sama), kembalikan rata-rata
        return round(np.mean(y))

    # Memprediksi nilai untuk x berikutnya
    next_x = len(y)
    forecast = m * next_x + b

    # Forecast tidak boleh negatif dan harus integer
    return max(0, round(forecast))


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
            # Jadikan Bay dan POD sebagai index dan ganti nama kolom Jumlah Kontainer
            renamed_df = df.set_index(["Bay", "Port of Discharge"])
            renamed_df = renamed_df.rename(columns={"Jumlah Kontainer": f"Jumlah ({name})"})
            dfs_to_merge.append(renamed_df)

    if not dfs_to_merge:
        return pd.DataFrame()

    # Menggabungkan semua DataFrame dengan outer join
    merged_df = pd.concat(dfs_to_merge, axis=1, join='outer')
    merged_df = merged_df.fillna(0).astype(int)

    # Menghitung kolom 'Forecast' menggunakan regresi linear per baris
    jumlah_cols = [col for col in merged_df.columns if col.startswith('Jumlah')]
    if jumlah_cols:
        merged_df['Forecast (Next Vessel)'] = merged_df[jumlah_cols].apply(forecast_next_value, axis=1).astype(int)

    # Mengurutkan berdasarkan Bay dan mengembalikan ke bentuk tabel datar
    merged_df = merged_df.reset_index()
    merged_df = merged_df.sort_values(by="Bay").reset_index(drop=True)

    return merged_df


def create_summary_table(pivots_dict):
    """
    Membuat tabel ringkasan dan menambahkan kolom forecast.
    """
    summaries = []
    for file_name, pivot in pivots_dict.items():
        if not pivot.empty:
            summary = pivot.groupby("Port of Discharge")["Jumlah Kontainer"].sum().reset_index()
            summary = summary.rename(columns={"Jumlah Kontainer": file_name})
            summary = summary.set_index("Port of Discharge")
            summaries.append(summary)
            
    if not summaries:
        return pd.DataFrame()

    # Gabungkan semua summary
    final_summary = pd.concat(summaries, axis=1).fillna(0).astype(int)
    
    # Hitung forecast untuk setiap POD jika data cukup
    if len(final_summary.columns) >= 2:
        final_summary['Forecast (Next Vessel)'] = final_summary.apply(forecast_next_value, axis=1).astype(int)
    
    # Tambahkan baris Total
    total_row = final_summary.sum().to_frame().T
    total_row.index = ["**TOTAL**"]
    # Pastikan tipe data total adalah integer
    final_summary = pd.concat([final_summary, total_row]).astype(int)
    
    return final_summary.reset_index()

def create_bay_allocation_table(comparison_df):
    """
    Membuat tabel alokasi forecast per Bay, memisahkan
    kontainer 20ft (Bay ganjil) dan 40ft (Bay genap).
    """
    if comparison_df.empty or 'Forecast (Next Vessel)' not in comparison_df.columns:
        return pd.DataFrame()

    df = comparison_df[['Bay', 'Port of Discharge', 'Forecast (Next Vessel)']].copy()
    
    # Hanya proses baris dengan forecast > 0
    df = df[df['Forecast (Next Vessel)'] > 0]
    
    if df.empty:
        return pd.DataFrame()

    # Tentukan tipe kontainer berdasarkan Bay ganjil/genap
    # Ganjil = 20, Genap = 40
    df['Container Type'] = np.where(df['Bay'] % 2 == 0, '40', '20')
    
    # Buat nama kolom tujuan (e.g., 'SGSIN 20')
    df['Allocation Column'] = df['Port of Discharge'] + ' ' + df['Container Type']

    # Lakukan pivot untuk membuat struktur tabel yang diinginkan
    allocation_pivot = df.pivot_table(
        index='Bay',
        columns='Allocation Column',
        values='Forecast (Next Vessel)',
        aggfunc='sum',
        fill_value=0
    )
    
    # Final formatting
    allocation_pivot = allocation_pivot.reset_index()
    
    return allocation_pivot

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

    # --- MENAMPILKAN RINGKASAN ---
    st.header("📊 Ringkasan Total Kontainer per Port of Discharge")
    summary_table = create_summary_table(pivots_dict)
    
    if not summary_table.empty:
        # Menerapkan perataan tengah pada tabel ringkasan
        st.dataframe(summary_table.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
    else:
        st.warning("Tidak ada data valid untuk membuat ringkasan.")
        
    st.markdown("---")

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
        
        # Membuat judul dinamis
        title = " vs ".join([f"`{name}`" for name in selected_files])
        st.subheader(f"Hasil: {title}")
        
        if not comparison_df.empty:
            # Menerapkan perataan tengah pada tabel perbandingan
            st.dataframe(comparison_df.style.set_properties(**{'text-align': 'center'}))

            # --- BUAT DAN TAMPILKAN TABEL ALOKASI BAY ---
            st.markdown("---")
            st.header("🎯 Prediksi Alokasi per Bay (20ft Ganjil / 40ft Genap)")
            allocation_table = create_bay_allocation_table(comparison_df)
            
            if not allocation_table.empty:
                st.dataframe(allocation_table.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
            else:
                st.info("Tidak ada data forecast untuk membuat tabel alokasi bay.")

        else:
            st.warning(f"Tidak dapat membandingkan file-file yang dipilih. Pastikan file valid dan berisi data dari POL 'IDJKT'.")
    else:
        st.warning("Silakan pilih minimal 2 file untuk ditampilkan perbandingannya.")
