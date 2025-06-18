import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="EDI Bay/POD Parser", layout="centered")

st.title("📦 EDI Container Bay & POD Analyzer")

uploaded_file = st.file_uploader("Upload file .EDI", type=["edi", "txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().splitlines()

    records = []
    
    # Variabel untuk menyimpan data kontainer saat ini yang sedang diproses
    current_container_data = {}
    
    # Flag untuk menunjukkan apakah kita sedang dalam "blok" data kontainer yang sedang dikumpulkan
    is_collecting_container_data = False

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

        # Kondisi untuk memulai pengumpulan data kontainer baru
        if line.startswith("LOC+147+"):
            # Jika kita sudah mengumpulkan data untuk kontainer sebelumnya, proses dulu
            if is_collecting_container_data and current_container_data:
                process_and_add_record(current_container_data)
            
            # Reset untuk kontainer baru dan mulai mengumpulkan
            current_container_data = {}
            is_collecting_container_data = True
            
            try:
                full_bay = line.split("+")[2].split(":")[0]
                # Menggunakan [0:2] untuk mengambil "02" dari "0221188"
                current_container_data["bay"] = full_bay[0:2] 
            except IndexError:
                st.warning(f"Format LOC+147+ tidak dikenal: {line}. Melewatkan Bay untuk kontainer ini.")
                current_container_data["bay"] = None 
        
        # Ekstrak Port of Discharge (POD) jika sedang dalam mode pengumpulan
        elif is_collecting_container_data and line.startswith("LOC+11+"):
            try:
                current_container_data["pod"] = line.split("+")[2].split(":")[0]
            except IndexError:
                st.warning(f"Format LOC+11+ tidak dikenal: {line}. Melewatkan POD untuk kontainer ini.")
                current_container_data["pod"] = None

        # Ekstrak Port of Loading (POL) jika sedang dalam mode pengumpulan
        elif is_collecting_container_data and line.startswith("LOC+9+"):
            try:
                current_container_data["pol"] = line.split("+")[2].split(":")[0]
            except IndexError:
                st.warning(f"Format LOC+9+ tidak dikenal: {line}. Melewatkan POL untuk kontainer ini.")
                current_container_data["pol"] = None

        # Kondisi untuk mengakhiri pengumpulan data kontainer (NAD+ atau EQD+CN+)
        # Berdasarkan file Anda, EQD+CN+ dan NAD+ muncul setelah semua LOC data kontainer
        # Kita bisa gunakan NAD+ sebagai penanda akhir yang lebih kuat,
        # atau EQD+CN+ yang menandai Container ID (tapi data LOC sudah di atasnya)
        # Mari kita gunakan NAD+ sebagai pemicu untuk memproses data kontainer sebelumnya
        elif is_collecting_container_data and line.startswith("NAD+"):
            # Jika NAD+ ditemukan, berarti data untuk kontainer yang sedang diproses sudah lengkap
            process_and_add_record(current_container_data)
            # Tidak perlu reset current_container_data atau is_collecting_container_data di sini,
            # karena LOC+147+ berikutnya yang akan memicu reset dan mulai baru.
            # Namun, jika tidak ada LOC+147+ setelah NAD+, data NAD+ tersebut tidak akan terproses.
            # Alternatif: Tetap set is_collecting_container_data = False dan current_container_data = {}
            # agar hanya LOC+147+ yang baru yang memulai pengumpulan.
            # Untuk skenario ini, kita akan membiarkan LOC+147+ berikutnya yang me-reset.

        # Jika kita melewati baris yang tidak relevan, biarkan saja
        # Jika Anda juga ingin mengambil EQD+CN+ (Container ID), bisa tambahkan di sini
        # elif is_collecting_container_data and line.startswith("EQD+CN+"):
        #     try:
        #         container_id = line.split("+")[2]
        #         current_container_data["container_id"] = container_id
        #     except IndexError:
        #         st.warning(f"Format EQD+CN+ tidak dikenal: {line}. Melewatkan Container ID.")
        #         current_container_data["container_id"] = None


    # Setelah loop selesai, pastikan memproses kontainer terakhir jika ada
    if is_collecting_container_data and current_container_data:
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
