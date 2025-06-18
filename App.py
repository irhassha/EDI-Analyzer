import streamlit as st
import pandas as pd
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
    return pivot_df


def compare_pivots(df1, df2, name1, name2):
    """
    Fungsi ini membandingkan dua DataFrame pivot dan mengembalikan
    DataFrame perbandingan beserta skor kesamaan.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame(), 0

    # Menjadikan Bay dan POD sebagai index untuk merge
    df1 = df1.set_index(["Bay", "Port of Discharge"])
    df2 = df2.set_index(["Bay", "Port of Discharge"])

    # Menggabungkan kedua DataFrame untuk perbandingan
    merged_df = pd.merge(
        df1,
        df2,
        left_index=True,
        right_index=True,
        how='outer',
        suffixes=(f' ({name1})', f' ({name2})')
    )
    # Mengisi data kosong dengan 0
    merged_df = merged_df.fillna(0)

    # Mengubah tipe data jumlah kontainer menjadi integer
    col1_name = f'Jumlah Kontainer ({name1})'
    col2_name = f'Jumlah Kontainer ({name2})'
    merged_df[col1_name] = merged_df[col1_name].astype(int)
    merged_df[col2_name] = merged_df[col2_name].astype(int)

    # Menghitung perbedaan dan status
    merged_df['Perbedaan'] = merged_df[col2_name] - merged_df[col1_name]

    def get_status(row):
        if row['Perbedaan'] == 0:
            return "Sama Persis"
        elif row[col1_name] == 0:
            return f"Baru"
        elif row[col2_name] == 0:
            return f"Hilang"
        elif row['Perbedaan'] > 0:
            return f"Bertambah"
        else:
            return f"Berkurang"

    merged_df['Status'] = merged_df.apply(get_status, axis=1)
    
    # Menghitung skor kesamaan
    total_combinations = len(merged_df)
    identical_combinations = len(merged_df[merged_df['Status'] == 'Sama Persis'])
    similarity_score = (identical_combinations / total_combinations) * 100 if total_combinations > 0 else 0
    
    return merged_df.reset_index(), similarity_score


def create_summary_table(pivots_dict):
    """
    Membuat tabel ringkasan jumlah kontainer per POD dari dictionary pivot table.
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
    
    # Tambahkan baris Total
    total_row = final_summary.sum().to_frame().T
    total_row.index = ["**TOTAL**"]
    final_summary = pd.concat([final_summary, total_row])

    return final_summary.reset_index()


# --- TAMPILAN APLIKASI STREAMLIT ---

st.title("🚢 EDI File Comparator")
st.caption("Unggah 2 file EDI atau lebih untuk membandingkan komposisi Bay, POD, dan jumlah kontainer.")

uploaded_files = st.file_uploader(
    "Upload file .EDI Anda di sini",
    type=["edi", "txt"],
    accept_multiple_files=True
)

if len(uploaded_files) < 2:
    st.info("ℹ️ Silakan unggah minimal 2 file untuk memulai perbandingan.")
else:
    # Memproses setiap file yang diunggah
    with st.spinner("Menganalisis file..."):
        pivots_dict = {f.name: parse_edi_to_pivot(f) for f in uploaded_files}

    # --- MENAMPILKAN RINGKASAN ---
    st.header("📊 Ringkasan Total Kontainer per Port of Discharge")
    summary_table = create_summary_table(pivots_dict)
    
    if not summary_table.empty:
        st.dataframe(summary_table, use_container_width=True)
    else:
        st.warning("Tidak ada data valid untuk membuat ringkasan.")
        
    st.markdown("---")

    # --- UI UNTUK MEMILIH FILE YANG AKAN DIBANDINGKAN ---
    st.header(f"🔍 Perbandingan Detail")
    
    file_names = list(pivots_dict.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.selectbox("Pilih file pertama (A):", file_names, index=0)
    with col2:
        # Membuat daftar pilihan untuk file kedua yang tidak sama dengan file pertama
        options_b = [name for name in file_names if name != file_a]
        if not options_b:
            st.warning("Hanya ada satu file, tidak bisa membandingkan.")
            st.stop()
        file_b = st.selectbox("Pilih file kedua (B):", options_b, index=0)

    # --- MENAMPILKAN HASIL PERBANDINGAN BERDASARKAN PILIHAN ---
    if file_a and file_b:
        pivot_a = pivots_dict[file_a]
        pivot_b = pivots_dict[file_b]
        
        comparison_df, score = compare_pivots(pivot_a, pivot_b, file_a, file_b)
        
        st.subheader(f"Hasil: `{file_a}` vs `{file_b}`")
        if not comparison_df.empty:
            st.metric(label="Tingkat Kesamaan (Bay, POD & Jumlah)", value=f"{score:.2f} %")
            st.dataframe(comparison_df)
        else:
            st.warning(f"Tidak dapat membandingkan `{file_a}` dan `{file_b}`. Pastikan kedua file valid dan berisi data dari POL 'IDJKT'.")

