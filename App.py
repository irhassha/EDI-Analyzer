import streamlit as st
import pandas as pd
import io # Import io for BytesIO

st.set_page_config(page_title="EDI Bay/POD Parser", layout="centered")

st.title("📦 EDI Container Bay & POD Analyzer")

uploaded_file = st.file_uploader("Upload file .EDI", type=["edi", "txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().splitlines()

    records = []
    
    # Inisialisasi variabel untuk menyimpan data kontainer saat ini
    # Kita akan mengisi ini, dan memprosesnya saat EQD+CN+ muncul
    current_container_data = {} 
    
    # Flag untuk menandai apakah kita sudah mulai mengumpulkan data kontainer yang valid
    # Ini membantu mengatasi data awal sebelum EQD+CN+ pertama
    first_container_data_started = False 

    def process_and_add_record(data_to_process):
        """Memproses data kontainer yang telah dikumpulkan dan menambahkannya ke records jika lengkap dan sesuai kriteria."""
        bay = data_to_process.get("bay")
        pod = data_to_process.get("pod")
        pol = data_to_process.get("pol")

        # Pastikan semua data ada dan Port of Loading adalah 'IDJKT'
        if bay and pod and pol == "IDJKT":
            records.append({
                "Bay": bay,
                "Port of Discharge": pod
            })

    for line in lines:
        line = line.strip()

        # Ketika segmen EQD+CN+ ditemukan, ini menandakan data kontainer sebelumnya selesai
        # Jadi, kita proses data yang sudah terkumpul untuk kontainer sebelumnya.
        if line.startswith("EQD+CN+"):
            if first_container_data_started: # Pastikan kita sudah mulai mengumpulkan data kontainer
                process_and_add_record(current_container_data)
            current_container_data = {} # Reset untuk kontainer baru
            first_container_data_started = True # Mulai mengumpulkan data untuk kontainer berikutnya

        # Ekstrak Bay
        elif line.startswith("LOC+147+"):
            try:
                full_bay = line.split("+")[2].split(":")[0]
                # Perhatikan: Sesuaikan indeks [1:3] jika Bay yang Anda inginkan adalah "02" dari "0221188"
                # Jika "02", pakai [0:2]. Jika "22", pakai [1:3] seperti semula.
                # Berdasarkan data Anda ("0221188"), saya asumsikan Anda ingin "02". Jadi saya ubah ke [0:2]
                current_container_data["bay"] = full_bay[0:2] 
            except IndexError:
                st.warning(f"Format LOC+147+ tidak dikenal: {line}. Melewatkan Bay.")
                current_container_data["bay"] = None 
            
        # Ekstrak Port of Discharge (POD)
        elif line.startswith("LOC+11+"):
            try:
                current_container_data["pod"] = line.split("+")[2].split(":")[0]
            except IndexError:
                st.warning(f"Format LOC+11+ tidak dikenal: {line}. Melewatkan POD.")
                current_container_data["pod"] = None

        # Ekstrak Port of Loading (POL)
        elif line.startswith("LOC+9+"):
            try:
                current_container_data["pol"] = line.split("+")[2].split(":")[0]
            except IndexError:
                st.warning(f"Format LOC+9+ tidak dikenal: {line}. Melewatkan POL.")
                current_container_data["pol"] = None

    # Penting: Setelah loop selesai, proses data kontainer terakhir jika ada dan sudah dimulai
    if first_container_data_started and current_container_data:
        process_and_add_record(current_container_data)

    if records:
        df = pd.DataFrame(records)

        st.subheader("📊 Tabel Kontainer")
        st.dataframe(df)

        st.subheader("🗁 Pivot: Jumlah Kontainer per Bay & Port")
        pivot_df = df.groupby(["Bay", "Port of Discharge"]).size().reset_index(name="Jumlah Kontainer")
        st.dataframe(pivot_df)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, index=False, sheet_name='Pivot Data')
        output_excel = output.getvalue()

        st.download_button(
            label="📅 Download Excel",
            data=output_excel,
            file_name="pivot_bay_pod.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("❗Tidak ditemukan data kontainer yang lengkap dari Port of Loading IDJKT (LOC+9+IDJKT) yang memiliki informasi Bay (LOC+147+) dan Port of Discharge (LOC+11+) dalam file.")
