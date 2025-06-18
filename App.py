import streamlit as st
import pandas as pd

st.set_page_config(page_title="EDI Bay/POD Parser", layout="centered")

st.title("📦 EDI Container Bay & POD Analyzer")

uploaded_file = st.file_uploader("Upload file .EDI", type=["edi", "txt"])

if uploaded_file is not None:
    # Baca isi file
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().splitlines()

    # Parsing blok kontainer
    records = []
    current_bay = None
    current_pod = None
    current_pol = None
    inside_container = False

    for line in lines:
        line = line.strip()

        if line.startswith("EQD+CN+"):
            inside_container = True
            current_bay = None
            current_pod = None
            current_pol = None

        elif inside_container and line.startswith("LOC+147+"):
            full_bay = line.split("+")[2].split(":")[0]
            current_bay = full_bay[1:3]  # Ambil dua digit setelah leading zero

        elif inside_container and line.startswith("LOC+11+"):
            current_pod = line.split("+")[2].split(":")[0]

        elif inside_container and line.startswith("LOC+9+"):
            current_pol = line.split("+")[2].split(":")[0]

        # Tambahkan jika ketiganya sudah lengkap
        if inside_container and current_bay and current_pod and current_pol:
            if current_pol == "IDJKT":
                records.append({
                    "Bay": current_bay,
                    "Port of Discharge": current_pod
                })
            inside_container = False
            current_bay = None
            current_pod = None
            current_pol = None

    if records:
        df = pd.DataFrame(records)

        st.subheader("📊 Tabel Kontainer")
        st.dataframe(df)

        st.subheader("🗁 Pivot: Jumlah Kontainer per Bay & Port")
        pivot_df = df.groupby(["Bay", "Port of Discharge"]).size().reset_index(name="Jumlah Kontainer")
        st.dataframe(pivot_df)

        # Download sebagai Excel
        output_excel = pivot_df.to_excel(index=False, engine='openpyxl')
        st.download_button(
            label="📅 Download Excel",
            data=output_excel,
            file_name="pivot_bay_pod.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("❗Tidak ditemukan data dari Port of Loading IDJKT dengan LOC+147 dan LOC+11 dalam file.")
