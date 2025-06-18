import streamlit as st
import pandas as pd

st.set_page_config(page_title="EDI Bay/POD Parser", layout="centered")

st.title("📦 EDI Container Bay & POD Analyzer")

uploaded_file = st.file_uploader("Upload file .EDI", type=["edi", "txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().splitlines()

    records = []
    
    # Inisialisasi variabel untuk menyimpan data kontainer saat ini
    current_container_data = {}

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

        # Ketika segmen EQD+CN+ ditemukan, ini menandakan awal kontainer baru.
        # Kita proses data kontainer sebelumnya (jika ada) dan reset untuk yang baru.
        if line.startswith("EQD+CN+"):
            if current_container_data: # Jika ada data dari kontainer sebelumnya
                process_and_add_record(current_container_data)
            current_container_data = {} # Reset untuk kontainer baru

        # Ekstrak Bay
        elif line.startswith("LOC+147+"):
            try:
                full_bay = line.split("+")[2].split(":")[0]
                current_container_data["bay"] = full_bay[1:3] # Ambil karakter ke-2 dan ke-3
            except IndexError:
                # Handle kasus jika format LOC+147+ tidak sesuai harapan
                st.warning(f"Format LOC+147+ tidak dikenal: {line}. Melewatkan Bay.")
                current_container_data["bay"] = None # Pastikan direset jika gagal
            
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

    # Penting: Setelah loop selesai, proses data kontainer terakhir jika ada
    if current_container_data:
        process_and_add_record(current_container_data)

    if records:
        df = pd.DataFrame(records)

        st.subheader("📊 Tabel Kontainer")
        st.dataframe(df)

        st.subheader("🗁 Pivot: Jumlah Kontainer per Bay & Port")
        # Menggunakan .value_counts() untuk ringkasan yang lebih cepat, lalu reset_index
        # atau tetap gunakan groupby seperti sebelumnya, keduanya baik
        pivot_df = df.groupby(["Bay", "Port of Discharge"]).size().reset_index(name="Jumlah Kontainer")
        st.dataframe(pivot_df)

        # Ubah data ke format Excel
        # Menggunakan BytesIO untuk menyimpan file di memori sebelum didownload
        import io
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
