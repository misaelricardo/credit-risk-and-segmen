from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import custom transformers dari file custom_transformers.py
import custom_transformers
from custom_transformers import GroupedImputerStage, RatioFeatureEngineer

# -------------------------------------------------------------
# FIX: Daftarkan class custom ke __main__ agar joblib bisa unpickle
# -------------------------------------------------------------
sys.modules["__main__"].GroupedImputerStage = GroupedImputerStage
sys.modules["__main__"].RatioFeatureEngineer = RatioFeatureEngineer
sys.modules["custom_transformers"] = custom_transformers

# -------------------------------------------------------------
# ACTION RULES MAPPING
# -------------------------------------------------------------
# Mapping Rule: (Cluster) -> (flag_utilisasi, flag_pendapatan, flag_skor_kredit) -> (Action Singkat, Action Detail)
ACTION_RULES = {
    0: {  # CLUSTER 0 — EARLY INTERVENTION
        (0, 0, 0): (
            "Pertahankan & Monitor",
            (
                "Pertahankan fasilitas berjalan dan lakukan monitoring rutin"
                " terhadap utilisasi, pendapatan, skor kredit, serta perubahan"
                " kondisi agunan. Tidak diperlukan intervensi khusus selama"
                " indikator tetap stabil."
            ),
        ),
        (0, 0, 1): (
            "Rehabilitasi Kredit",
            (
                "Tinjau riwayat pembayaran dan penyebab skor kredit rendah."
                " Batasi penambahan exposure yang tidak diperlukan dan tetapkan"
                " target perbaikan perilaku pembayaran."
            ),
        ),
        (0, 1, 0): (
            "Penyesuaian Pembayaran",
            (
                "Evaluasi cash flow dan kemampuan bayar. Jika terdapat mismatch"
                " antara pendapatan dan kewajiban, sesuaikan jadwal atau"
                " cicilan agar lebih sesuai dengan kemampuan pembayaran."
            ),
        ),
        (0, 1, 1): (
            "Restrukturisasi Terarah",
            (
                "Lakukan review kondisi keuangan, identifikasi penyebab"
                " penurunan pendapatan, susun repayment plan, dan batasi"
                " tambahan exposure. Pertimbangkan restrukturisasi jika"
                " kemampuan bayar tidak lagi memadai."
            ),
        ),
        (1, 0, 0): (
            "Kurangi Utilisasi",
            (
                "Kurangi ketergantungan pada revolving balance, tetapkan batas"
                " utilisasi yang lebih sehat, dan dorong percepatan pembayaran"
                " saldo kartu kredit."
            ),
        ),
        (1, 0, 1): (
            "Konsolidasi Utang",
            (
                "Konsolidasikan kewajiban revolving yang memenuhi syarat menjadi"
                " skema pembayaran terstruktur, batasi tambahan unsecured"
                " exposure, dan tetapkan target pembayaran."
            ),
        ),
        (1, 1, 0): (
            "Restrukturisasi Cash Flow",
            (
                "Hitung kembali cicilan yang sustainable, kurangi exposure"
                " revolving, dan sesuaikan repayment schedule agar beban"
                " pembayaran lebih sesuai dengan kemampuan cash flow."
            ),
        ),
        (1, 1, 1): (
            "Evaluasi Restrukturisasi Dini",
            (
                "Lakukan penilaian restrukturisasi secara proaktif. Jika cash"
                " flow masih viable, pertimbangkan tenor lebih pendek dengan"
                " penyesuaian bunga sementara serta pengendalian exposure,"
                " sesuai kebijakan bank."
            ),
        ),
    },
    1: {  # CLUSTER 1 — INTENSIVE INTERVENTION
        (0, 0, 0): (
            "Enhanced Monitoring",
            (
                "Pertahankan fasilitas jika performance stabil, tetapi lakukan"
                " credit reassessment lebih berkala dan monitoring kondisi"
                " agunan lebih ketat karena cluster memiliki tingkat"
                " kerentanan lebih tinggi."
            ),
        ),
        (0, 0, 1): (
            "Containment & Remediasi",
            (
                "Batasi incremental exposure, review limit/facility, evaluasi"
                " penyebab skor rendah, dan buat repayment milestones dengan"
                " monitoring yang lebih sering."
            ),
        ),
        (0, 1, 0): (
            "Evaluasi Restrukturisasi",
            (
                "Evaluasi kemampuan bayar berbasis cash flow dan hitung"
                " sustainable installment. Jika tekanan pendapatan berlanjut,"
                " pertimbangkan penyesuaian repayment schedule."
            ),
        ),
        (0, 1, 1): (
            "Restrukturisasi Intensif",
            (
                "Lakukan financial review, exposure containment, repayment"
                " plan, dan monitoring intensif. Restrukturisasi"
                " dipertimbangkan bila masih terdapat kemampuan bayar yang"
                " sustainable."
            ),
        ),
        (1, 0, 0): (
            "Reduksi Utilisasi",
            (
                "Kurangi atau batasi fasilitas revolving tambahan, ubah saldo"
                " revolving yang eligible menjadi structured installment, dan"
                " lakukan monitoring utilisasi secara berkala."
            ),
        ),
        (1, 0, 1): (
            "Konsolidasi & Kurangi Exposure",
            (
                "Konsolidasikan utang yang eligible, batasi kredit baru,"
                " strukturkan repayment, dan lakukan reassessment terhadap"
                " limit serta fasilitas yang berjalan."
            ),
        ),
        (1, 1, 0): (
            "Stabilisasi Cash Flow",
            (
                "Lakukan detailed cash-flow assessment, kurangi revolving"
                " exposure, sesuaikan jadwal pembayaran, dan pertimbangkan"
                " pengurangan fasilitas bila repayment capacity tidak memadai."
            ),
        ),
        (1, 1, 1): (
            "Workout / Restrukturisasi Intensif",
            (
                "Prioritaskan untuk review. Jika masih viable, lakukan"
                " structured restructuring dengan exposure control. Jika tidak"
                " viable, arahkan ke recovery/workout strategy sesuai kebijakan"
                " dan kewenangan."
            ),
        ),
    },
}


def get_action_recommendation(
    cluster, flag_utilisasi, flag_pendapatan, flag_skor_kredit
):
  try:
    c = int(cluster)
    u = int(flag_utilisasi)
    p = int(flag_pendapatan)
    s = int(flag_skor_kredit)
    return ACTION_RULES.get(c, {}).get(
        (u, p, s), ("Action Tidak Ditemukan", "Kombinasi parameter tidak valid")
    )
  except Exception:
    return ("N/A", "Data flag tidak lengkap.")


# -------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# -------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment - Properti",
    page_icon="🏦",
    layout="wide",
)


# -------------------------------------------------------------
# 2. LOAD MODEL PIPELINE & DATASET CSV
# -------------------------------------------------------------
@st.cache_resource
def load_pipeline():
  return joblib.load("model_gagal_bayar.joblib")


try:
  pipeline = load_pipeline()
except Exception as e:
  st.error(f"Gagal memuat model 'model_gagal_bayar.joblib': {e}")
  st.info(
      "Pastikan file 'model_gagal_bayar.joblib' dan 'custom_transformers.py'"
      " berada di direktori yang sama dengan 'app.py'."
  )
  st.stop()

# Path dataset CSV untuk tab pencarian debitur
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "hasil_prediksi_cluster_gagal_bayar (1).csv"


@st.cache_data
def load_debitur_dataset():
  if CSV_PATH.exists():
    return pd.read_csv(CSV_PATH)
  elif (BASE_DIR / "hasil_prediksi_cluster_gagal_bayar (1).csv").exists():
    return pd.read_csv(BASE_DIR / "hasil_prediksi_cluster_gagal_bayar (1).csv")
  else:
    raise FileNotFoundError(f"File CSV tidak ditemukan di `{CSV_PATH}`")


# -------------------------------------------------------------
# 3. HEADER & NAVIGASI TAB
# -------------------------------------------------------------
st.title("🏦 Sistem Analisis Risiko & Segmentasi Debitur Properti")
st.markdown("""
Aplikasi evaluasi profil kredit: lakukan **Prediksi Risiko Baru (Probability of Default)** atau **Cari & Tinjau Profil Debitur Eksisting**.
""")
st.divider()

# Pembagian Tab Navigasi
tab_prediksi, tab_detail = st.tabs(
    ["🚀 Form Prediksi Risiko Baru", "🔍 Pencarian Detail Profil Debitur"]
)


# =============================================================
# TAB 1: FORM PREDIKSI BARU
# =============================================================
with tab_prediksi:
  st.subheader("📋 Input Parameter Calon Debitur")

  with st.form("debitur_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
      st.markdown("#### 👤 Profil Finansial & Pekerjaan")
      debitur_id = st.text_input("ID Debitur", value="PBI-NEW001")
      pendapatan_bulanan = st.number_input(
          "Pendapatan Bulanan (Rp)",
          min_value=0.0,
          value=25000000.0,
          step=1000000.0,
          format="%.0f",
          help="Isi 0 jika tidak ada data (akan diimputasi otomatis oleh model)",
      )
      status_pekerjaan = st.selectbox(
          "Status Pekerjaan", ["Karyawan Tetap", "Wiraswasta", "Profesional"]
      )
      jumlah_tanggungan = st.number_input(
          "Jumlah Tanggungan (Orang)",
          min_value=0,
          max_value=15,
          value=1,
          step=1,
      )
      metode_pembayaran = st.selectbox(
          "Metode Pembayaran",
          ["Auto-Debit", "Virtual Account", "Transfer Manual"],
      )

    with col2:
      st.markdown("#### 💳 Pinjaman & Histori Kredit")
      skor_kredit_internal = st.number_input(
          "Skor Kredit Internal",
          min_value=300,
          max_value=850,
          value=650,
          step=1,
          help="Skor riwayat kredit (rentang: 300 - 850)",
      )
      utilisasi_kartu_kredit = st.number_input(
          "Utilisasi Kartu Kredit (%)",
          min_value=0.0,
          max_value=100.0,
          value=30.0,
          step=0.5,
          format="%.2f",
          help="Persentase pemakaian batas limit kartu kredit (0 - 100%)",
      )
      jumlah_tunggakan_historis = st.number_input(
          "Jumlah Tunggakan Historis (Kali)",
          min_value=0,
          max_value=30,
          value=0,
          step=1,
      )
      plafon_pinjaman = st.number_input(
          "Plafon Pinjaman (Rp)",
          min_value=50000000.0,
          value=750000000.0,
          step=25000000.0,
          format="%.0f",
      )
      tenor_bulan = st.selectbox(
          "Tenor Pinjaman (Bulan)", [60, 84, 120, 180, 240], index=3
      )

    with col3:
      st.markdown("#### 🏠 Karakteristik Agunan")
      nilai_agunan_awal = st.number_input(
          "Nilai Agunan Awal (Rp)",
          min_value=50000000.0,
          value=1000000000.0,
          step=25000000.0,
          format="%.0f",
      )
      nilai_agunan_kini = st.number_input(
          "Nilai Agunan Terkini (Rp)",
          min_value=0.0,
          value=950000000.0,
          step=25000000.0,
          format="%.0f",
          help=(
              "Isi 0 jika belum ada penilaian terbaru (akan diimputasi otomatis)"
          ),
      )
      wilayah_agunan = st.selectbox(
          "Wilayah Agunan",
          [
              "Jakarta Selatan",
              "Bogor",
              "Depok",
              "Tangerang (BSD/Serpong)",
              "Bekasi",
          ],
      )
      subsegmen_properti = st.selectbox(
          "Subsegmen Properti",
          ["Rumah Tapak", "Apartemen", "Tanah Kavling", "Ruko-Komersial"],
      )
      jenis_sertifikat = st.selectbox("Jenis Sertifikat", ["SHM", "HGB"])
      usia_bangunan_tahun = st.number_input(
          "Usia Bangunan (Tahun)",
          min_value=0.0,
          max_value=60.0,
          value=5.0,
          step=1.0,
          help="Isi 0 jika tidak diketahui (akan diimputasi otomatis)",
      )

    btn_submit = st.form_submit_button(
        "🔍 Prediksi & Hitung PD", use_container_width=True
    )

  if btn_submit:
    agunan_kini_val = np.nan if nilai_agunan_kini <= 0 else nilai_agunan_kini
    pendapatan_val = np.nan if pendapatan_bulanan <= 0 else pendapatan_bulanan
    usia_bangunan_val = (
        np.nan if usia_bangunan_tahun <= 0 else usia_bangunan_tahun
    )

    df_input = pd.DataFrame([{
        "debitur_id": debitur_id,
        "skor_kredit_internal": skor_kredit_internal,
        "subsegmen_properti": subsegmen_properti,
        "wilayah_agunan": wilayah_agunan,
        "nilai_agunan_awal": nilai_agunan_awal,
        "nilai_agunan_kini": agunan_kini_val,
        "plafon_pinjaman": plafon_pinjaman,
        "tenor_bulan": tenor_bulan,
        "pendapatan_bulanan": pendapatan_val,
        "jumlah_tanggungan": jumlah_tanggungan,
        "status_pekerjaan": status_pekerjaan,
        "jumlah_tunggakan_historis": jumlah_tunggakan_historis,
        "utilisasi_kartu_kredit": utilisasi_kartu_kredit,
        "metode_pembayaran": metode_pembayaran,
        "usia_bangunan_tahun": usia_bangunan_val,
        "jenis_sertifikat": jenis_sertifikat,
    }])

    try:
      raw_pd = pipeline.predict_proba(df_input)[0, 1]
      pred_class = pipeline.predict(df_input)[0]

      df_imputed = pipeline.named_steps["imputer"].transform(df_input)
      df_featured = pipeline.named_steps["feature_engineer"].transform(
          df_imputed
      )

      ltv_awal_val = df_featured["ltv_awal"].iloc[0] * 100
      ltv_kini_val = df_featured["ltv_kini"].iloc[0] * 100
      penurunan_val = df_featured["persentase_penurunan"].iloc[0] * 100
      dti_val = df_featured["dti_ratio"].iloc[0] * 100

      agunan_kini_real = df_featured["nilai_agunan_kini"].iloc[0]
      pendapatan_real = df_featured["pendapatan_bulanan"].iloc[0]
      usia_bangunan_real = df_featured["usia_bangunan_tahun"].iloc[0]

      st.subheader("🎯 Hasil Prediksi Model")
      res1, res2, res3 = st.columns([1.2, 1.2, 1.3])

      with res1:
        st.metric(
            label="Probability of Default (Raw Sigmoid PD)",
            value=f"{raw_pd:.4f}",
            help=(
                "Probabilitas gagal bayar murni dari output sigmoid Logistic"
                " Regression"
            ),
        )
        st.caption(f"Persentase: **{raw_pd * 100:.2f}%**")

      with res2:
        st.markdown("**Hasil Prediksi:**")
        if pred_class == 1:
          st.markdown(
              "<h3 style='color: #FF4B4B; margin: 0;'>⚠️ GAGAL BAYAR</h3>",
              unsafe_allow_html=True,
          )
          st.caption("Prediksi: Debitur berisiko wanprestasi / macet.")
        else:
          st.markdown(
              "<h3 style='color: #09AB3B; margin: 0;'>✅ TIDAK GAGAL"
              " BAYAR</h3>",
              unsafe_allow_html=True,
          )
          st.caption("Prediksi: Debitur diprediksi lancar.")

      with res3:
        st.markdown("**📊 Rasio & Evaluasi Finansial:**")
        st.write(f"• **LTV Awal:** `{ltv_awal_val:.2f}%`")
        st.write(f"• **LTV Kini:** `{ltv_kini_val:.2f}%`")

        if nilai_agunan_kini <= 0:
          st.write(
              f"• **Penurunan Nilai Agunan:** `{penurunan_val:.2f}%` *(Imputasi:"
              f" Rp {agunan_kini_real:,.0f})*"
          )
        else:
          st.write(f"• **Penurunan Nilai Agunan:** `{penurunan_val:.2f}%`")

        if pendapatan_bulanan <= 0:
          st.write(
              f"• **DTI Ratio:** `{dti_val:.2f}%` *(Imputasi Pendapatan: Rp"
              f" {pendapatan_real:,.0f})*"
          )
        else:
          st.write(f"• **DTI Ratio:** `{dti_val:.2f}%`")

        if usia_bangunan_tahun <= 0:
          st.write(
              f"• **Usia Bangunan:** `{usia_bangunan_real:.0f} Tahun` *(Imputasi"
              " Wilayah)*"
          )

      st.divider()

      df_result = df_input.copy()
      df_result["raw_probability_of_default"] = raw_pd
      df_result["prediction_class"] = pred_class

      csv_data = df_result.to_csv(index=False).encode("utf-8")

      st.markdown("#### 📥 Ekspor Hasil Prediksi")
      st.download_button(
          label="Download Data & Nilai PD (CSV)",
          data=csv_data,
          file_name=f"credit_risk_result_{debitur_id}.csv",
          mime="text/csv",
          help=(
              "Unduh data input beserta nilai Probability of Default (PD) dalam"
              " format CSV"
          ),
      )
      st.dataframe(df_result)

    except Exception as err:
      st.error(f"Terjadi kesalahan saat memproses inferensi: {err}")


# =============================================================
# TAB 2: PENCARIAN & DETAIL DEBITUR EKSISTING
# =============================================================
with tab_detail:
  st.subheader("🔎 Profil & Informasi Historis Debitur")
  st.caption(
      "Lihat ringkasan parameter finansial, hasil evaluasi risiko (PD/Sigmoid),"
      " segmentasi cluster, serta rekomendasi aksi."
  )

  try:
    df_debitur = load_debitur_dataset()
    daftar_id = df_debitur["debitur_id"].dropna().unique().tolist()

    selected_id = st.selectbox(
        "Pilih atau Cari ID Debitur:", options=daftar_id, index=0
    )

    if selected_id:
      row = df_debitur[df_debitur["debitur_id"] == selected_id].iloc[0]

      # ---------------------------------------------------------
      # 1. Ambil Hasil Prediksi & Probabilitas Sigmoid dari CSV
      # ---------------------------------------------------------
      raw_pd_exist = row.get("y_pred_proba (sigmoid)", None)
      pred_class_exist = row.get("y_pred (dibulatkan, threshold=0.5)", None)

      cluster_val = row.get("cluster", 0)
      flag_skor = row.get("flag_skor_kredit", 0)
      flag_util = row.get("flag_utilisasi", 0)
      flag_pend = row.get("flag_pendapatan", 0)

      action_singkat, action_detail = get_action_recommendation(
          cluster=cluster_val,
          flag_utilisasi=flag_util,
          flag_pendapatan=flag_pend,
          flag_skor_kredit=flag_skor,
      )

      if raw_pd_exist is not None and pd.notna(raw_pd_exist):
        raw_pd_float = float(raw_pd_exist)
        pred_class_int = (
            int(pred_class_exist)
            if pd.notna(pred_class_exist)
            else (1 if raw_pd_float >= 0.5 else 0)
        )

        st.markdown("---")
        st.markdown("#### 🎯 Evaluasi Risiko & Rekomendasi")
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.3])

        with res_col1:
          st.metric(
              label="Probability of Default (Raw Sigmoid PD)",
              value=f"{raw_pd_float:.4f}",
              help="Nilai output sigmoid probabilitas gagal bayar dari model",
          )
          st.caption(f"Persentase: **{raw_pd_float * 100:.2f}%**")

        with res_col2:
          st.markdown("**Hasil Prediksi:**")
          if pred_class_int == 1:
            st.markdown(
                "<h3 style='color: #FF4B4B; margin: 0;'>⚠️ GAGAL BAYAR</h3>",
                unsafe_allow_html=True,
            )
            st.caption("Status: Debitur berisiko wanprestasi / macet.")
          else:
            st.markdown(
                "<h3 style='color: #09AB3B; margin: 0;'>✅ TIDAK GAGAL"
                " BAYAR</h3>",
                unsafe_allow_html=True,
            )
            st.caption("Status: Debitur diprediksi lancar.")

        with res_col3:
          cluster_desc = (
              "Early Intervention" if int(cluster_val) == 0 else "Intensive"
          )
          st.markdown(f"**📊 Segmentasi:** `Cluster {cluster_val}`")
          st.caption(f"Karakteristik: {cluster_desc}")
          st.metric(
              label="⚡ Action Singkat",
              value=action_singkat,
              help=f"Detail Tindakan:\n{action_detail}",
          )

      st.markdown("---")

      # ---------------------------------------------------------
      # 2. Ringkasan Parameter Pinjaman & Agunan
      # ---------------------------------------------------------
      plafon = row.get("plafon_pinjaman", 0)
      agunan_awal = row.get("nilai_agunan_awal", 0)
      agunan_kini = row.get("nilai_agunan_kini", 0)
      tenor = row.get("tenor_bulan", 0)
      pendapatan = row.get("pendapatan_bulanan", 0)
      wilayah = row.get("wilayah_agunan", "-")

      c1, c2, c3 = st.columns(3)
      with c1:
        st.metric(
            label="Plafon Pinjaman",
            value=f"Rp {float(plafon):,.0f}" if pd.notna(plafon) else "Rp 0",
        )
        st.metric(
            label="Tenor Pinjaman",
            value=f"{int(tenor)} Bulan" if pd.notna(tenor) else "-",
        )

      with c2:
        st.metric(
            label="Nilai Agunan Awal",
            value=(
                f"Rp {float(agunan_awal):,.0f}"
                if pd.notna(agunan_awal)
                else "Rp 0"
            ),
        )
        st.metric(
            label="Nilai Agunan Kini",
            value=(
                f"Rp {float(agunan_kini):,.0f}"
                if pd.notna(agunan_kini) and float(agunan_kini) > 0
                else "Belum Dinilai"
            ),
        )

      with c3:
        st.metric(
            label="Pendapatan Bulanan",
            value=(
                f"Rp {float(pendapatan):,.0f}"
                if pd.notna(pendapatan) and float(pendapatan) > 0
                else "Tidak Ada Data"
            ),
        )
        st.metric(label="Wilayah Agunan", value=f"{wilayah}")

      st.markdown("---")

      # ---------------------------------------------------------
      # 3. Tabel Detail Lengkap Debitur
      # ---------------------------------------------------------
      st.markdown("#### 📋 Ringkasan Parameter Debitur")

      # Format Rasio LTV & DTI dari CSV (dikonversi ke float agar aman dari string)
      ltv_awal_raw = row.get("ltv_awal", None)
      ltv_kini_raw = row.get("ltv_kini", None)
      dti_raw = row.get("dti_ratio", None)

      ltv_awal_str = (
          f"{float(ltv_awal_raw)*100:.2f}%" if pd.notna(ltv_awal_raw) else "-"
      )
      ltv_kini_str = (
          f"{float(ltv_kini_raw)*100:.2f}%" if pd.notna(ltv_kini_raw) else "-"
      )
      dti_str = f"{float(dti_raw)*100:.2f}%" if pd.notna(dti_raw) else "-"

      data_ringkasan = {
          "Parameter": [
              "ID Debitur",
              "Probability of Default (Sigmoid PD)",
              "Prediksi Status",
              "Cluster",
              "Rekomendasi Action Singkat",
              "Plafon Pinjaman",
              "Nilai Agunan Awal",
              "Nilai Agunan Terkini",
              "LTV Awal / LTV Kini",
              "DTI Ratio",
              "Tenor Pinjaman",
              "Pendapatan Bulanan",
              "Wilayah Agunan",
          ],
          "Nilai": [
              str(selected_id),
              (
                  f"{raw_pd_float:.4f} ({raw_pd_float*100:.2f}%)"
                  if pd.notna(raw_pd_exist)
                  else "-"
              ),
              "Gagal Bayar" if pred_class_int == 1 else "Tidak Gagal Bayar",
              f"Cluster {cluster_val}",
              action_singkat,
              f"Rp {float(plafon):,.0f}" if pd.notna(plafon) else "-",
              f"Rp {float(agunan_awal):,.0f}" if pd.notna(agunan_awal) else "-",
              (
                  f"Rp {float(agunan_kini):,.0f}"
                  if pd.notna(agunan_kini) and float(agunan_kini) > 0
                  else "-"
              ),
              f"{ltv_awal_str} / {ltv_kini_str}",
              dti_str,
              (
                  f"{int(tenor)} Bulan ({int(tenor)/12:.1f} Tahun)"
                  if pd.notna(tenor) and tenor > 0
                  else "-"
              ),
              (
                  f"Rp {float(pendapatan):,.0f}"
                  if pd.notna(pendapatan) and float(pendapatan) > 0
                  else "-"
              ),
              str(wilayah),
          ],
      }
      st.table(pd.DataFrame(data_ringkasan))

  except FileNotFoundError as fnf_err:
    st.warning(
        f"File dataset CSV belum ditemukan: {fnf_err}. Pastikan file berada di"
        " folder `data/` atau satu folder dengan `app.py`."
    )
  except Exception as e:
    st.error(f"Gagal memuat data debitur: {e}")