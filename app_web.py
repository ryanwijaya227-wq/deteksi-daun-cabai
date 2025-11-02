import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os
import gdown

# -----------------------
# CONFIG / STYLE
# -----------------------
st.set_page_config(page_title="Deteksi & Pengolahan Citra Daun Cabai", layout="wide")

st.markdown("""
<style>
body { background-color: #0f1720; color: #e6eef8; }
.big-title { font-size:40px; font-weight:700; color:#22d3ee; }
.card { background:#0b1220; border-radius:12px; padding:18px; box-shadow:0 6px 18px rgba(0,0,0,0.4); }
.result { border-radius:10px; padding:12px; background:#071022; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown('<div class="big-title">🌶️ Deteksi Penyakit & Pengolahan Citra Daun Cabai</div>', unsafe_allow_html=True)

# -----------------------
# Download model otomatis (kalau belum ada)
# -----------------------
MODEL_PATH = "model_cabai.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1zZRZkwuLAAnPFon4WytBxUegM4N_jWTC"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Mengunduh model dari Google Drive..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model berhasil diunduh!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['antraknosa', 'daun_keriting', 'bercak_daun', 'busuk_daun', 'sehat', 'virus_mozaik']
display_names = {
    'antraknosa': ('Antraknosa', '#f97316'),
    'daun_keriting': ('Daun Keriting', '#f43f5e'),
    'bercak_daun': ('Bercak Daun', '#f43f5e'),
    'busuk_daun': ('Busuk Daun', '#ef4444'),
    'sehat': ('Sehat', '#10b981'),
    'virus_mozaik': ('Virus Mozaik', '#f59e0b'),
}

# -----------------------
# Upload area
# -----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload gambar daun cabai...", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# When file is uploaded
# -----------------------
if uploaded_file is not None:
    # Original Image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Asli (RGB)", use_column_width=True)

    # -------------------------------
    # ✳️ Tahapan Pengolahan Citra
    # -------------------------------
    st.subheader("🔍 Tahapan Pengolahan Citra Digital")

    col1, col2, col3 = st.columns(3)

    # 1️⃣ Grayscale (Pengolahan Citra Berwarna)
    with col1:
        gray = ImageOps.grayscale(img)
        st.image(gray, caption="Grayscale Image", use_column_width=True)
        st.caption("➡️ Mengubah citra RGB menjadi skala abu-abu (Pertemuan 5).")

    # 2️⃣ Binerisasi (Pengolahan Citra Biner)
    with col2:
        bw = gray.point(lambda x: 0 if x < 120 else 255, '1')
        st.image(bw, caption="Citra Biner (Thresholding)", use_column_width=True)
        st.caption("➡️ Mengubah citra menjadi hitam-putih untuk segmentasi (Pertemuan 6).")

    # 3️⃣ Enhancement (Perbaikan Citra)
    with col3:
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(1.5)
        st.image(enhanced, caption="Citra Hasil Enhancement", use_column_width=True)
        st.caption("➡️ Meningkatkan kontras agar fitur daun lebih jelas (Pertemuan 3 & 4).")

    st.markdown("---")

    # -------------------------------
    # 🔮 PREPROCESSING untuk Model CNN
    # -------------------------------
    img_resized = img.resize((150, 150))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0) / 255.0

    # -------------------------------
    # 🤖 PREDIKSI MODEL
    # -------------------------------
    with st.spinner("Model sedang memproses gambar..."):
        preds = model.predict(x)
    probs = tf.nn.softmax(preds[0]).numpy()
    idx = int(np.argmax(probs))
    cls = class_names[idx]
    label_text, color = display_names.get(cls, (cls, "#9ca3af"))
    confidence = float(probs[idx]) * 100

    st.markdown(f"""
    <div class="result">
        <h3 style="color:{color}; margin:0;">🌿 Prediksi: <strong>{label_text}</strong></h3>
        <p style="color:#cbd5e1; margin:4px 0;">Tingkat Keyakinan: <strong>{confidence:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    st.progress(int(confidence))

    # Tabel probabilitas semua kelas
    st.write("📊 **Probabilitas Tiap Kelas:**")
    st.table({class_names[i]: [f"{probs[i]*100:.2f}%"] for i in range(len(class_names))})

    # -------------------------------
    # 🩺 Saran Berdasarkan Hasil
    # -------------------------------
    if cls == 'sehat':
        st.success("✅ Daun tampak SEHAT. Pertahankan perawatan tanaman secara rutin.")
    else:
        st.error(f"⚠️ Daun terdeteksi **{label_text}**. Disarankan lakukan pemeriksaan lebih lanjut atau pengendalian penyakit.")

    st.markdown("---")
