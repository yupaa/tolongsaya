import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
from io import BytesIO
from pathlib import Path


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Monthly Tracker PDB Sektoral",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# NAMA SEKTOR
# ============================================================

nama_sektor = {
    "A": "Pertanian, Kehutanan, dan Perikanan",
    "B": "Pertambangan dan Penggalian",
    "C": "Industri Pengolahan",
    "D": "Pengadaan Listrik dan Gas",
    "E": "Pengadaan Air, Pengelolaan Sampah, Limbah dan Daur Ulang",
    "F": "Konstruksi",
    "G": "Perdagangan Besar dan Eceran",
    "H": "Transportasi dan Pergudangan",
    "I": "Penyediaan Akomodasi dan Makan Minum",
    "J": "Informasi dan Komunikasi",
    "K": "Jasa Keuangan dan Asuransi",
    "L": "Real Estat",
    "MN": "Jasa Perusahaan",
    "O": "Administrasi Pemerintahan",
    "P": "Jasa Pendidikan",
    "Q": "Jasa Kesehatan dan Kegiatan Sosial",
    "RSTU": "Jasa Lainnya"
}

urutan_sektor = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "MN", "O", "P", "Q", "RSTU"
]


# ============================================================
# VARIABEL STATISTIK RESMI
# ============================================================

official_features = [
    "inflasi",
    "jisdor",
    "import",
    "export"
]


# ============================================================
# 43 VARIABEL GOOGLE TRENDS
# ============================================================

gt_features = [
    "Perkebunan",
    "Peternakan",
    "Hortikultura",
    "Perburuan",
    "Perikanan",

    "Gas alam",
    "Minyak bumi",
    "Energi panas bumi",

    "Industri",
    "Industri makanan",
    "Industri tekstil",
    "Industri kimia",
    "Farmasi industri",
    "Industri pulp dan kertas",
    "Perabotan",

    "Listrik",
    "Es batu",

    "Pengelolaan sampah",
    "Pengolahan limbah",
    "Penyediaan air",
    "Daur ulang",

    "Konstruksi",

    "Perkulakan",
    "Perdagangan",

    "Transportasi",

    "Akomodasi",
    "Makanan dan minuman",

    "Komunikasi",
    "Informasi",
    "Telekomunikasi",
    "Penerbitan",

    "Asuransi",
    "Jasa keuangan",

    "Lahan yasan",

    "Konsultan",
    "Alih daya",

    "Pengadaan",
    "Pemerintahan",

    "Kesehatan",
    "Pendidikan",
    "Pekerja sosial",

    "Hiburan",
    "Kesenian"
]


# ============================================================
# CEK JUMLAH GT
# ============================================================

if len(gt_features) != 43:
    st.warning(
        f"Daftar Google Trends berisi {len(gt_features)} variabel, "
        "bukan 43."
    )


# ============================================================
# SELURUH KOLOM INPUT
# ============================================================

template_columns = (
    ["Periode"]
    + official_features
    + gt_features
)


# ============================================================
# LOAD MODEL
# ============================================================

# Path dibuat relatif terhadap app.py, bukan terhadap folder tempat
# perintah `streamlit run` dijalankan.
MODEL_PATH = Path(__file__).parent / "model_terbaik.joblib"


@st.cache_resource
def load_models(path, mtime):
    # mtime ikut jadi kunci cache: kalau file model diganti,
    # model otomatis dimuat ulang.
    return joblib.load(path)


try:
    model_dashboard = load_models(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )

except Exception as e:
    st.error("❌ File model_terbaik.joblib tidak dapat dimuat.")
    st.error(f"Detail error: {e}")
    st.stop()


# ============================================================
# VALIDASI MODEL
# ============================================================

sektor_model = list(model_dashboard.keys())

sektor_hilang = [
    sektor
    for sektor in urutan_sektor
    if sektor not in model_dashboard
]

if len(sektor_hilang) > 0:

    st.error(
        "❌ Model untuk beberapa sektor tidak ditemukan: "
        + ", ".join(sektor_hilang)
    )

    st.stop()


# Sektor S3/S5 wajib punya objek PCA di dalam file model
sektor_pca_bermasalah = [
    s for s in urutan_sektor
    if model_dashboard[s].get("skenario") in ["S3", "S5"]
    and not all(
        k in model_dashboard[s]
        for k in ["pca", "pca_scaler", "fitur_pca_input"]
    )
]

if len(sektor_pca_bermasalah) > 0:

    st.error(
        "❌ File model yang terbaca tidak memuat objek PCA untuk sektor: "
        + ", ".join(sektor_pca_bermasalah)
        + ". Kemungkinan file model_terbaik.joblib yang dipakai "
        "adalah versi lama. Ganti dengan file terbaru lalu reboot app."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Monthly Tracker")

    st.markdown("---")

    # ========================================================
    # MENU
    # ========================================================

    st.markdown("### Menu")

    if st.button("🏠 Beranda", use_container_width=True):
        st.session_state.halaman = "Beranda"

    if st.button("📈 Monthly Tracker", use_container_width=True):
        st.session_state.halaman = "Monthly Tracker"

    # Default halaman
    if "halaman" not in st.session_state:
        st.session_state.halaman = "Beranda"

    st.markdown("---")

    # ========================================================
    # INFORMASI DASHBOARD
    # ========================================================

    st.write(
        f"**Jumlah sektor:** {len(model_dashboard)}"
    )

    st.write(
        f"**Google Trends:** {len(gt_features)} variabel"
    )

    st.write(
        f"**Variabel makro:** {len(official_features)} variabel"
    )

    st.caption(
        f"Model: {MODEL_PATH.stat().st_size:,} byte | "
        f"scikit-learn {sklearn.__version__}"
    )

    st.markdown("---")

    st.caption(
        "Dashboard nowcasting pertumbuhan PDB "
        "17 sektor ekonomi."
    )


# ============================================================
# HALAMAN AKTIF
# ============================================================

halaman = st.session_state.halaman

# ============================================================
# HALAMAN BERANDA
# ============================================================

if halaman == "Beranda":

    st.title(
        "📊 Monthly Tracker Pertumbuhan PDB "
        "17 Sektor Ekonomi"
    )

    st.markdown(
        """
        ### Tentang Dashboard

        Dashboard ini digunakan untuk menghasilkan *monthly tracker*
        pertumbuhan PDB pada 17 sektor ekonomi di Indonesia dengan
        memanfaatkan indikator berfrekuensi bulanan.

        Model yang digunakan merupakan model terbaik yang telah
        diperoleh melalui proses pemodelan pada penelitian.
        Setiap sektor memiliki model dan skenario prediktor terbaik
        masing-masing.

        ### Cara Menggunakan

        1. Siapkan data indikator bulanan sesuai dengan template.
        2. Masuk ke halaman **Monthly Tracker**.
        3. Upload file Excel.
        4. Klik **Lakukan Prediksi**.
        5. Dashboard akan menghasilkan estimasi pertumbuhan PDB
           untuk seluruh 17 sektor.

        Model dan skenario prediktor tidak perlu dipilih secara manual.
        Dashboard akan menggunakan model terbaik yang telah ditetapkan
        untuk masing-masing sektor.
        """
    )

    st.markdown("---")

    st.subheader("17 Sektor Ekonomi")

    sektor_display = pd.DataFrame({
        "Kode": urutan_sektor,
        "Sektor": [
            nama_sektor[s]
            for s in urutan_sektor
        ]
    })

    st.dataframe(
        sektor_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HALAMAN MONTHLY TRACKER
# ============================================================

elif halaman == "Monthly Tracker":

    st.title("📈 Monthly Tracker Pertumbuhan PDB")

    st.markdown(
        """
        Upload data indikator bulanan dalam format Excel.
        Dashboard akan menggunakan model terbaik masing-masing sektor
        untuk menghasilkan estimasi pertumbuhan PDB.
        """
    )


    # ========================================================
    # TEMPLATE EXCEL
    # ========================================================

    st.subheader("1. Template Input")

    st.write(
        """
        Gunakan template berikut agar nama variabel sesuai
        dengan model yang digunakan.
        """
    )

    st.info(
        "Semua variabel (statistik resmi dan Google Trends) harus diisi "
        "dalam bentuk **pertumbuhan y-o-y (%)**, sama seperti data "
        "yang dipakai saat training, bukan nilai mentah."
    )


    # Data kosong untuk template
    template_df = pd.DataFrame(
        columns=template_columns
    )


    # Petunjuk
    petunjuk_data = []

    petunjuk_data.append({
        "Variabel": "Periode",
        "Jenis": "Identitas",
        "Keterangan":
            "Periode bulanan, misalnya 2025M01, 2025M02, dan seterusnya."
    })

    for fitur in official_features:

        petunjuk_data.append({
            "Variabel": fitur,
            "Jenis": "Statistik resmi",
            "Keterangan":
                f"Pertumbuhan y-o-y (%) {fitur} pada bulan tersebut."
        })

    for fitur in gt_features:

        petunjuk_data.append({
            "Variabel": fitur,
            "Jenis": "Google Trends",
            "Keterangan":
                f"Pertumbuhan y-o-y (%) Google Trends untuk topik {fitur}."
        })


    petunjuk_df = pd.DataFrame(
        petunjuk_data
    )


    # Buat Excel
    template_buffer = BytesIO()

    with pd.ExcelWriter(
        template_buffer,
        engine="openpyxl"
    ) as writer:

        template_df.to_excel(
            writer,
            index=False,
            sheet_name="Data"
        )

        petunjuk_df.to_excel(
            writer,
            index=False,
            sheet_name="Petunjuk"
        )

    template_buffer.seek(0)


    st.download_button(
        label="⬇️ Download Template Excel",
        data=template_buffer,
        file_name="template_monthly_tracker.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


    # ========================================================
    # UPLOAD
    # ========================================================

    st.subheader("2. Upload Data")

    uploaded_file = st.file_uploader(
        "Upload file Excel",
        type=["xlsx", "xls"]
    )


    if uploaded_file is not None:

        # ====================================================
        # BACA EXCEL
        # ====================================================

        try:

            excel_file = pd.ExcelFile(
                uploaded_file
            )

            # Prioritaskan sheet Data
            if "Data" in excel_file.sheet_names:

                df = pd.read_excel(
                    uploaded_file,
                    sheet_name="Data"
                )

            else:

                df = pd.read_excel(
                    uploaded_file,
                    sheet_name=excel_file.sheet_names[0]
                )

        except Exception as e:

            st.error(
                "❌ File Excel tidak dapat dibaca."
            )

            st.error(str(e))

            st.stop()


        # ====================================================
        # CEK DATA KOSONG
        # ====================================================

        if df.empty:

            st.error(
                "❌ File Excel tidak memiliki data."
            )

            st.stop()


        st.success(
            "✅ File berhasil dibaca."
        )


        # ====================================================
        # CEK KOLOM
        # ====================================================

        kolom_hilang = [
            col
            for col in template_columns
            if col not in df.columns
        ]

        kolom_tambahan = [
            col
            for col in df.columns
            if col not in template_columns
        ]


        if len(kolom_hilang) > 0:

            st.error(
                "❌ File belum sesuai dengan template."
            )

            st.write(
                "**Kolom yang belum tersedia:**"
            )

            st.write(kolom_hilang)

            st.stop()


        if len(kolom_tambahan) > 0:

            st.warning(
                "⚠️ Terdapat kolom tambahan yang "
                "tidak digunakan."
            )

            st.write(kolom_tambahan)


        # ====================================================
        # CEK TIPE DATA
        # ====================================================

        numeric_columns = (
            official_features
            + gt_features
        )

        kolom_non_numerik = []

        for col in numeric_columns:

            if not pd.api.types.is_numeric_dtype(
                df[col]
            ):

                kolom_non_numerik.append(col)


        if len(kolom_non_numerik) > 0:

            st.error(
                "❌ Variabel berikut harus berupa angka:"
            )

            st.write(
                kolom_non_numerik
            )

            st.stop()


        # ====================================================
        # CEK MISSING VALUE
        # ====================================================

        missing = (
            df[numeric_columns]
            .isna()
            .sum()
        )

        missing = missing[
            missing > 0
        ]


        if len(missing) > 0:

            st.error(
                "❌ Terdapat nilai kosong pada data."
            )

            st.dataframe(
                missing.rename(
                    "Jumlah Missing Value"
                ),
                use_container_width=True
            )

            st.stop()


        # ====================================================
        # PREVIEW
        # ====================================================

        st.subheader("Preview Data")

        st.dataframe(
            df.head(),
            use_container_width=True,
            hide_index=True
        )

        st.write(
            f"Jumlah periode: **{len(df)}**"
        )


        # ====================================================
        # TOMBOL PREDIKSI
        # ====================================================

        st.subheader("3. Prediksi")

        prediksi_button = st.button(
            "🔮 Lakukan Prediksi",
            type="primary",
            use_container_width=True
        )


        if prediksi_button:

            hasil_tracker = {}   # {kode sektor: prediksi seluruh periode}
            info_sektor = []
            error_list = []


            # =================================================
            # LOOP 17 SEKTOR
            # =================================================

            for sektor in urutan_sektor:

                info_model = model_dashboard[sektor]

                metode = info_model.get("metode")
                skenario = info_model.get("skenario")

                try:

                    model = info_model["model"]
                    scaler_model = info_model["scaler"]
                    fitur = info_model["fitur"]


                    # -----------------------------------------
                    # S1, S2, S4, S6, S7
                    # -----------------------------------------

                    if skenario in ["S1", "S2", "S4", "S6", "S7"]:

                        fitur_hilang = [
                            f for f in fitur
                            if f not in df.columns
                        ]

                        if len(fitur_hilang) > 0:
                            raise ValueError(
                                "Fitur model tidak ditemukan "
                                "dalam file Excel: "
                                + ", ".join(fitur_hilang)
                            )

                        X_input = df[fitur].copy()


                    # -----------------------------------------
                    # S3 / S5
                    # -----------------------------------------

                    elif skenario in ["S3", "S5"]:

                        pca = info_model.get("pca")
                        pca_scaler = info_model.get("pca_scaler")
                        fitur_pca_input = info_model.get(
                            "fitur_pca_input"
                        )

                        if pca is None or pca_scaler is None:
                            raise ValueError(
                                f"{skenario} membutuhkan objek PCA dan "
                                "scaler PCA, tetapi tidak ada di file "
                                "model (kemungkinan file model versi lama)."
                            )

                        if fitur_pca_input is None:
                            raise ValueError(
                                f"Fitur input PCA untuk {skenario} "
                                "tidak ditemukan di file model."
                            )

                        fitur_hilang = [
                            f for f in fitur_pca_input
                            if f not in df.columns
                        ]

                        if len(fitur_hilang) > 0:
                            raise ValueError(
                                "Fitur Google Trends untuk PCA tidak "
                                "ditemukan dalam file Excel: "
                                + ", ".join(fitur_hilang)
                            )

                        X_GT = df[fitur_pca_input].copy()
                        X_GT_scaled = pca_scaler.transform(X_GT)
                        X_pca = pca.transform(X_GT_scaled)

                        pc_names = info_model.get("pc_names") or [
                            f"PC{i}"
                            for i in range(1, X_pca.shape[1] + 1)
                        ]

                        X_pca = pd.DataFrame(
                            X_pca,
                            columns=pc_names,
                            index=df.index
                        )

                        if skenario == "S3":
                            X_input = X_pca[fitur].copy()

                        else:  # S5
                            X_input = pd.concat(
                                [X_pca, df[official_features]],
                                axis=1
                            )[fitur].copy()


                    else:

                        raise ValueError(
                            f"Skenario {skenario} tidak dikenali."
                        )


                    # =========================================
                    # SCALING MODEL + PREDIKSI (SEMUA PERIODE)
                    # =========================================

                    X_scaled = scaler_model.transform(X_input)

                    prediksi = model.predict(X_scaled)

                    hasil_tracker[sektor] = np.round(
                        np.asarray(prediksi, dtype=float),
                        2
                    )

                    info_sektor.append({
                        "Kode Sektor": sektor,
                        "Sektor": nama_sektor[sektor],
                        "Model": metode,
                        "Skenario": skenario,
                        "CV RMSE": round(
                            float(info_model["CV RMSE"]), 4
                        )
                    })


                except Exception as e:

                    error_list.append({
                        "Kode Sektor": sektor,
                        "Sektor": nama_sektor[sektor],
                        "Model": metode,
                        "Skenario": skenario,
                        "Masalah": f"{type(e).__name__}: {e}"
                    })


            # =================================================
            # HASIL
            # =================================================

            if len(hasil_tracker) > 0:

                st.markdown("---")

                st.subheader("4. Hasil Monthly Tracker")

                st.success(
                    f"✅ Prediksi berhasil dilakukan untuk "
                    f"{len(hasil_tracker)} sektor dan "
                    f"{len(df)} periode."
                )

                df_tracker = pd.DataFrame({
                    "Periode": df["Periode"].values
                })

                for sektor in urutan_sektor:
                    if sektor in hasil_tracker:
                        df_tracker[sektor] = hasil_tracker[sektor]

                df_info = pd.DataFrame(info_sektor)

                st.write(
                    "**Prediksi pertumbuhan PDB y-o-y (%) per periode**"
                )

                st.dataframe(
                    df_tracker,
                    use_container_width=True,
                    hide_index=True
                )

                st.write("**Model yang digunakan per sektor**")

                st.dataframe(
                    df_info,
                    use_container_width=True,
                    hide_index=True
                )


                # =============================================
                # DOWNLOAD HASIL
                # =============================================

                hasil_buffer = BytesIO()

                with pd.ExcelWriter(
                    hasil_buffer,
                    engine="openpyxl"
                ) as writer:

                    df_tracker.to_excel(
                        writer,
                        index=False,
                        sheet_name="Tracker"
                    )

                    df_info.to_excel(
                        writer,
                        index=False,
                        sheet_name="Info Model"
                    )

                hasil_buffer.seek(0)

                st.download_button(
                    label="⬇️ Download Hasil Prediksi",
                    data=hasil_buffer,
                    file_name="hasil_monthly_tracker.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    )
                )


            # =================================================
            # ERROR
            # =================================================

            if len(error_list) > 0:

                st.markdown("---")

                st.warning(
                    "⚠️ Beberapa sektor belum dapat diprediksi."
                )

                st.dataframe(
                    pd.DataFrame(error_list),
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dashboard Monthly Tracker Pertumbuhan PDB 17 Sektor Ekonomi"
)