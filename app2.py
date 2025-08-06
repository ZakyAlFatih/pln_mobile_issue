import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# --- Konfigurasi Halaman dan Tema ---
st.set_page_config(
    page_title="Analisis Sentimen Ulasan",
    layout="wide"
)

# Tema kustom
st.markdown("""
<style>
    .main {
        background-color: #F0F8FF;
    }
    .stApp {
        background: linear-gradient(135deg, #87CEEB, #FFFACD);
    }
    h1, h2, h3 {
        color: #00008B;
    }
    body, .markdown-text-container, .stText, .stMarkdown, .stDataFrame, .stTable, .stSubheader {
        color: #000000 !important;
    }
    .stButton > button {
        background-color: #FFD700;
        color: #00008B;
        border: 1px solid #00008B;
        font-weight: bold;
    }
    .stTextInput > div > div > textarea, .stTextInput > div > div > input {
        border: 2px solid #00008B;
        background-color: #FFFFFF;
        color: #000000;
    }
    .stFileUploader > div > button {
        background-color: #FFFACD;
        border: 2px dashed #00008B;
        color: #00008B;
    }
</style>
""", unsafe_allow_html=True)

# --- Komponen Notifikasi Kustom ---
def notify_success(text):
    st.markdown(f"""
    <div style='
        background-color:#d4edda;
        color:#155724;
        padding:10px;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        font-weight: bold;
    '>
    ✅ {text}
    </div>
    """, unsafe_allow_html=True)

def notify_error(text):
    st.markdown(f"""
    <div style='
        background-color:#f8d7da;
        color:#721c24;
        padding:10px;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
        font-weight: bold;
    '>
    ❌ {text}
    </div>
    """, unsafe_allow_html=True)

def notify_warning(text):
    st.markdown(f"""
    <div style='
        background-color:#fff3cd;
        color:#856404;
        padding:10px;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        font-weight: bold;
    '>
    ⚠️ {text}
    </div>
    """, unsafe_allow_html=True)

# --- Fungsi untuk konversi Excel ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- Load Model ---
try:
    model, label_encoder = joblib.load("svm2_pipeline.pkl")
    notify_success("Model berhasil dimuat!")
except FileNotFoundError:
    notify_error("File 'svm_pipeline.pkl' tidak ditemukan. Pastikan file ada di direktori yang benar.")
    st.stop()
except Exception as e:
    notify_error(f"Gagal memuat model. Error: {e}")
    notify_warning("Pastikan file .pkl Anda berisi tuple dengan dua elemen (model, label_encoder).")
    st.stop()

# --- Tampilan Aplikasi ---
st.title("Analisis Keluhan User pada Aplikasi PLN Mobile 📝")
st.write("Aplikasi ini menggunakan model SVM untuk memprediksi keluhan dari teks ulasan.")
st.markdown("---")

# --- Prediksi Teks Tunggal ---
st.markdown("<h2>1. Prediksi Teks Tunggal</h2>", unsafe_allow_html=True)
text_input = st.text_area("Masukkan teks ulasan Anda di sini:", height=150)

if st.button("Prediksi Sentimen Teks"):
    if text_input:
        with st.spinner('Menganalisis...'):
            prediction_numeric = model.predict([text_input])
            prediction_label = label_encoder.inverse_transform(prediction_numeric)
            st.markdown("<h3 style='color:#00008B;'>Hasil Prediksi:</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px; color:#000000;'><strong>Label Prediksi:</strong> {prediction_label[0]}</p>", unsafe_allow_html=True)
    else:
        notify_warning("Mohon masukkan teks ulasan terlebih dahulu.")

st.markdown("---")

# --- Prediksi dari File Excel ---
st.markdown("<h2>2. Prediksi dari File Excel</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Unggah file Excel Anda. Pastikan ada kolom bernama **'ISI ULASAN'**.",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        if "ISI ULASAN" not in df.columns:
            notify_error("Kolom 'ISI ULASAN' tidak ditemukan dalam file yang diunggah.")
        else:
            st.write("**Pratinjau Data:**")
            st.dataframe(df.head())

            if st.button("Proses File dan Prediksi Semua Ulasan"):
                with st.spinner('Memproses semua ulasan dalam file...'):
                    df_clean = df.dropna(subset=['ISI ULASAN'])
                    df_clean = df_clean[df_clean['ISI ULASAN'].str.strip() != '']

                    predictions_numeric = model.predict(df_clean['ISI ULASAN'])
                    predictions_label = label_encoder.inverse_transform(predictions_numeric)

                    df_clean['PREDIKSI_SENTIMEN'] = predictions_label

                    notify_success("Prediksi selesai!")
                    st.write("**Hasil Prediksi:**")
                    st.dataframe(df_clean.head())

                    excel_data = to_excel(df_clean)
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi (Excel)",
                        data=excel_data,
                        file_name="hasil_prediksi_sentimen.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    except Exception as e:
        notify_error(f"Terjadi kesalahan saat memproses file: {e}")
