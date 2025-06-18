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
    df_filtered = df_all[df_all["Port of Loading"] == "IDJKT"].copy()

    if df_filtered.empty:
        return pd.DataFrame()
        
    df_display = df_filtered.drop(columns=["Port of Loading"]).dropna(subset=["Port of Discharge", "Bay"])
    
    if df_display.empty:
        return pd.DataFrame()

    # Membuat pivot table
    pivot_df = df_display.groupby(["Bay", "Port of Discharge"]).size().reset_index(name="Jumlah Kontainer")
    return pivot_df.set_index(["Bay", "Port of Discharge"])


def compare_pivots(df1, df2, name1, name2):
    """
    Fungsi ini membandingkan dua DataFrame pivot dan mengembalikan
    DataFrame perbandingan beserta skor kesamaan.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame(), 0

    # Menggabungkan kedua DataFrame untuk perbandingan
    merged_df = pd.merge(
        df1,
        df2,
        left_index=True,
        right_index=True,
        how='outer',
        suffixes=(f'_{name1}', f'_{name2}')
    )
    # Mengisi data kosong dengan 0
    merged_df = merged_df.fillna(0)

    # Mengubah tipe data jumlah kontainer menjadi integer
    col1_name = f'Jumlah Kontainer_{name1}'
    col2_name = f'Jumlah Kontainer_{name2}'
    merged_df[col1_name] = merged_df[col1_name].astype(int)
    merged_df[col2_name] = merged_df[col2_name].astype(int)

    # Menghitung perbedaan dan status
    merged_df['Perbedaan'] = merged_df[col2_name] - merged_df[col1_name]

    def get_status(row):
        if row['Perbedaan'] == 0:
            return "Sama Persis"
        elif row[col1_name] == 0:
            return f"Baru di {name2}"
        elif row[col2_name] == 0:
            return f"Hilang di {name2}"
        elif row['Perbedaan'] > 0:
            return f"Bertambah di {name2}"
        else:
            return f"Berkurang di {name2}"

    merged_df['Status'] = merged_df.apply(get_status, axis=1)
    
    # Menghitung skor kesamaan
    total_combinations = len(merged_df)
    identical_combinations = len(merged_df[merged_df['Status'] == 'Sama Persis'])
    similarity_score = (identical_combinations / total_combinations) * 100 if total_combinations > 0 else 0
    
    return merged_df.reset_index(), similarity_score


# --- TAMPILAN APLIKASI STREAMLIT ---

st.title("🚢 EDI File Comparator")
st.caption("Unggah 3 file EDI untuk membandingkan komposisi Bay, POD, dan jumlah kontainer.")

uploaded_files = st.file_uploader(
    "Upload 3 file .EDI di sini",
    type=["edi", "txt"],
    accept_multiple_files=True
)

if len(uploaded_files) != 3:
    st.info("ℹ️ Silakan unggah tepat 3 file untuk memulai perbandingan.")
else:
    st.success("✅ 3 file berhasil diunggah. Memproses perbandingan...")

    # Memproses setiap file yang diunggah
    with st.spinner("Menganalisis file..."):
        pivot1 = parse_edi_to_pivot(uploaded_files[0])
        pivot2 = parse_edi_to_pivot(uploaded_files[1])
        pivot3 = parse_edi_to_pivot(uploaded_files[2])

    file_names = [f.name for f in uploaded_files]

    # Melakukan dan menampilkan perbandingan
    st.header(f"Perbandingan: `{file_names[0]}` vs `{file_names[1]}`")
    comparison_1_2, score_1_2 = compare_pivots(pivot1, pivot2, "File1", "File2")
    
    if not comparison_1_2.empty:
        st.metric(label="Tingkat Kesamaan (Bay, POD & Jumlah)", value=f"{score_1_2:.2f} %")
        st.dataframe(comparison_1_2)
    else:
        st.warning("Tidak dapat membandingkan file 1 dan 2. Pastikan kedua file valid dan berisi data dari POL 'IDJKT'.")

    st.header(f"Perbandingan: `{file_names[1]}` vs `{file_names[2]}`")
    comparison_2_3, score_2_3 = compare_pivots(pivot2, pivot3, "File2", "File3")

    if not comparison_2_3.empty:
        st.metric(label="Tingkat Kesamaan (Bay, POD & Jumlah)", value=f"{score_2_3:.2f} %")
        st.dataframe(comparison_2_3)
    else:
        st.warning("Tidak dapat membandingkan file 2 dan 3. Pastikan kedua file valid dan berisi data dari POL 'IDJKT'.")

