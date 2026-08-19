import streamlit as st
import pandas as pd
from pathlib import Path

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Detail Debitur", page_icon="🔍", layout="wide")

# 2. Path File Dataset CSV
# Mengambil root folder project (satu tingkat di atas folder /pages)
BASE_DIR = Path(__file__).resolve().parent.parent

# Menyusun path: root_project/data/hasil_prediksi_cluster_gagal_bayar (1).csv
CSV_PATH = BASE_DIR / "data" / "hasil_prediksi_cluster_gagal_bayar (1).csv"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"File dataset tidak ditemukan di: `{CSV_PATH}`. Pastikan folder `data` dan nama file CSV sudah sesuai.")
    st.stop()

# 3. Header & Dropdown Pilihan Debitur ID
st.title("🔍 Profil & Detail Informasi Debitur")
st.caption("Pilih ID Debitur untuk melihat detail plafon, agunan, tenor, pendapatan, dan wilayah agunan.")

daftar_id = df["debitur_id"].dropna().unique().tolist()
selected_id = st.selectbox(
    "Pilih atau Cari Debitur ID:",
    options=daftar_id,
    index=0 if len(daftar_id) > 0 else None
)

if selected_id:
    # Filter data debitur terpilih
    row = df[df["debitur_id"] == selected_id].iloc[0]

    # Ambil nilai fitur
    plafon = row.get("plafon_pinjaman", 0)
    agunan_awal = row.get("nilai_agunan_awal", 0)
    agunan_kini = row.get("nilai_agunan_kini", 0)
    tenor = row.get("tenor_bulan", 0)
    pendapatan = row.get("pendapatan_bulanan", 0)
    wilayah = row.get("wilayah_agunan", "-")

    st.divider()

    # 4. Tampilan Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Plafon Pinjaman", value=f"Rp {plafon:,.0f}")
        st.metric(label="Tenor Pinjaman", value=f"{tenor} Bulan")

    with col2:
        st.metric(label="Nilai Agunan Awal", value=f"Rp {agunan_awal:,.0f}")
        st.metric(
            label="Nilai Agunan Kini",
            value=f"Rp {agunan_kini:,.0f}" if pd.notna(agunan_kini) and agunan_kini > 0 else "Belum Dinilai"
        )

    with col3:
        st.metric(
            label="Pendapatan Bulanan",
            value=f"Rp {pendapatan:,.0f}" if pd.notna(pendapatan) and pendapatan > 0 else "Tidak Ada Data"
        )
        st.metric(label="Wilayah Agunan", value=f"{wilayah}")

    st.divider()

    # 5. Tampilan Tabel Ringkasan
    st.subheader("📋 Ringkasan Parameter")

    data_ringkasan = {
        "Parameter": [
            "ID Debitur",
            "Plafon Pinjaman",
            "Nilai Agunan Awal",
            "Nilai Agunan Terkini",
            "Tenor Pinjaman",
            "Pendapatan Bulanan",
            "Wilayah Agunan"
        ],
        "Nilai": [
            str(selected_id),
            f"Rp {plafon:,.0f}",
            f"Rp {agunan_awal:,.0f}",
            f"Rp {agunan_kini:,.0f}" if pd.notna(agunan_kini) and agunan_kini > 0 else "-",
            f"{tenor} Bulan ({tenor/12:.1f} Tahun)" if tenor > 0 else "-",
            f"Rp {pendapatan:,.0f}" if pd.notna(pendapatan) and pendapatan > 0 else "-",
            str(wilayah)
        ]
    }

    st.table(pd.DataFrame(data_ringkasan))