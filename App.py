import streamlit as st
import pandas as pd
import io # Diperlukan untuk proses download file Excel
import openpyxl

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="EDI Bay/POD Parser", layout="centered")

# Judul aplikasi
st.title("📦 EDI Container Bay & POD Analyzer")

# Komponen untuk mengunggah file
uploaded_file = st.file_uploader("Upload file .EDI Anda", type=["edi", "txt"])

# Blok utama yang akan berjalan jika file sudah diunggah
if uploaded_file is not None:
    # Membaca dan mendekode isi file
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().splitlines()

    all_records = []
    current_container_data = {} # Menggunakan dictionary untuk menampung data kontainer saat ini

    # Iterasi melalui setiap baris dalam file
    for line in lines:
        line = line.strip()

        # Jika menemukan LOC+147, ini adalah penanda kontainer baru.
        # Simpan data kontainer sebelumnya jika ada, lalu mulai yang baru.
        if line.startswith("LOC+147+"):
            if current_container_data:  # Jika ada data di penampung, simpan dulu
                all_records.append(current_container_data)
            
            # Reset untuk kontainer baru dan ambil data Bay
            current_container_data = {}
            try:
                # Mengambil kode Bay (misal: 01, 03, 05 dari B01, B03, B05)
                full_bay = line.split("+")[2].split(":")[0]
                current_container_data["Bay"] = full_bay[1:3] if len(full_bay) >= 3 else full_bay
            except IndexError:
                current_container_data["Bay"] = None

        # Kumpulkan data lain untuk kontainer yang sedang diproses
        elif line.startswith("LOC+11+"):  # Port of Discharge
            try:
                # Menghapus apostrof jika ada
                pod = line.split("+")[2].split(":")[0]
                current_container_data["Port of Discharge"] = pod.replace("'", "")
            except IndexError:
                pass  # Abaikan jika format baris salah

        elif line.startswith("LOC+9+"):  # Port of Loading
            try:
                # Membersihkan spasi, mengubah ke huruf besar, DAN MENGHAPUS APOSTROF
                pol = line.split("+")[2].split(":")[0]
                current_container_data["Port of Loading"] = pol.strip().upper().replace("'", "")
            except IndexError:
                pass

    # Jangan lupa simpan data kontainer terakhir setelah loop selesai
    if current_container_data:
        all_records.append(current_container_data)

    # Proses data jika berhasil mengumpulkan record
    if all_records:
        # Konversi semua data yang terkumpul ke DataFrame Pandas
        df_all = pd.DataFrame(all_records)
        
        st.subheader("✅ Semua Data Kontainer yang Ditemukan")
        st.caption("Gunakan tabel ini untuk memeriksa semua data yang berhasil dibaca dari file sebelum difilter.")
        st.dataframe(df_all)

        # ---- TAHAP FILTERING ----
        # Sekarang, filter DataFrame untuk Port of Loading 'IDJKT'
        df_filtered = df_all[df_all["Port of Loading"] == "IDJKT"].copy()
        
        # Hapus kolom yang tidak relevan lagi setelah difilter
        if not df_filtered.empty:
            # Hapus kolom Port of Loading karena kita sudah tahu itu IDJKT
            # dan hapus baris yang tidak memiliki data POD atau Bay
            df_display = df_filtered.drop(columns=["Port of Loading"]).dropna(subset=["Port of Discharge", "Bay"])
        else:
            df_display = df_filtered

        # Tampilkan hasil jika data yang difilter tidak kosong
        if not df_display.empty:
            st.success(f"Ditemukan {len(df_display)} kontainer dari Port of Loading IDJKT.")
            st.subheader("📊 Tabel Kontainer dari POL: IDJKT")
            st.dataframe(df_display)

            st.subheader("🗂️ Pivot: Jumlah Kontainer per Bay & POD")
            pivot_df = df_display.groupby(["Bay", "Port of Discharge"]).size().reset_index(name="Jumlah Kontainer")
            st.dataframe(pivot_df)
            
            # --- Proses untuk membuat file Excel dan tombol download ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pivot_df.to_excel(writer, index=False, sheet_name='Pivot Bay POD')
            
            st.download_button(
                label="📥 Download Data Pivot (Excel)",
                data=output.getvalue(),
                file_name="pivot_bay_pod_idjkt.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            # Pesan error yang lebih spesifik jika tidak ada data IDJKT
            st.warning("✔️ File berhasil dibaca, namun tidak ditemukan data dengan Port of Loading 'IDJKT'.")
            st.info("Tips: Periksa tabel 'Semua Data Kontainer yang Ditemukan' di atas untuk melihat kode Port of Loading apa saja yang ada di dalam file Anda.")

    else:
        # Pesan error jika file tidak mengandung data yang bisa dibaca
        st.error("❗ Gagal mem-parsing data dari file. Pastikan file EDI mengandung segmen `LOC+147`.")
