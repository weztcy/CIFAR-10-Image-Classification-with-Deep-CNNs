"""
Streamlit Web Application for CIFAR-10 Image Classification with Deep CNNs.
Provides interactive image upload, sample gallery, model switching,
real-time prediction, probability breakdown, and architecture details.
"""

import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import CIFARPredictor
from data_loader import load_cifar_raw

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classification | Deep CNNs",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .pred-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .pred-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
    }
    .pred-class {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .pred-conf {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor(model_type='resnet'):
    checkpoint_file = f"outputs/checkpoints/{model_type}_best.pt"
    return CIFARPredictor(checkpoint_path=checkpoint_file, model_type=model_type)


@st.cache_data
def get_sample_cifar_images():
    """Load 1 sample for each of the 10 classes from test_batch."""
    try:
        _, _, test_images, test_labels, class_names = load_cifar_raw('dataset')
        samples = {}
        for idx, name in enumerate(class_names):
            match_idx = np.where(test_labels == idx)[0][0]
            samples[name] = Image.fromarray(test_images[match_idx])
        return samples, class_names
    except Exception as e:
        return {}, []


def main():
    st.markdown('<div class="main-header">CIFAR-10 Image Classification with Deep CNNs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Aplikasi inferensi interaktif klasifikasi citra 10 kelas menggunakan PyTorch Deep Convolutional Neural Networks</div>', unsafe_allow_html=True)

    # Query Parameters handling
    query_model = st.query_params.get("model", "resnet")
    if query_model not in ["resnet", "baseline"]:
        query_model = "resnet"
    
    query_view = st.query_params.get("view", "sample")
    query_sample = st.query_params.get("sample", "frog")

    # Sidebar: Controls & Metadata
    with st.sidebar:
        st.header("⚙️ Konfigurasi Model")
        model_options = ["resnet", "baseline"]
        model_choice = st.selectbox(
            "Pilih Arsitektur Model:",
            options=model_options,
            index=0 if query_model == "resnet" else 1,
            format_func=lambda x: "ResNetCIFAR (Unggulan - 80.34% Acc)" if x == "resnet" else "Baseline CNN (667K params - 77.78% Acc)"
        )

        st.markdown("---")
        st.header("📊 Ringkasan Model Terpilih")
        if model_choice == "resnet":
            st.markdown("""
            - **Arsitektur:** ResNetCIFAR (Deep Residual CNN)
            - **Stem:** 3×3 Conv (Stride 1, No MaxPool)
            - **Head:** Global Average Pooling + Linear
            - **Parameter:** 175,258 (Ringan & Cepat)
            - **Test Accuracy:** **80.34%**
            - **Macro F1-Score:** **0.8027**
            - **Ukuran Checkpoint:** 2.08 MB
            - **Latency CPU:** 2.88 ms / citra
            """)
        else:
            st.markdown("""
            - **Arsitektur:** Baseline CNN (VGG-style 3-Stage)
            - **Stem:** 3×3 Conv + MaxPool (2×2)
            - **Head:** Flatten + Dense (2048 → 256 → 10)
            - **Parameter:** 667,690
            - **Test Accuracy:** 77.78%
            - **Macro F1-Score:** 0.7742
            - **Ukuran Checkpoint:** 7.68 MB
            - **Latency CPU:** 2.02 ms / citra
            """)

        st.markdown("---")
        st.header("ℹ️ Karakteristik Dataset")
        st.caption("""
        Model dilatih pada citra CIFAR-10 berukuran **32 × 32 piksel**. 
        Foto eksternal beresolusi tinggi akan di-*downscale* ke 32×32 piksel sebelum inferensi. 
        Perbedaan domain (pencahayaan, latar belakang kompleks) dapat mempengaruhi akurasi prediksi.
        """)

    # Load Predictor
    try:
        predictor = get_predictor(model_choice)
    except Exception as e:
        st.error(f"Gagal memuat checkpoint model: {e}")
        st.stop()

    samples, class_names = get_sample_cifar_images()

    # Image Input Selection
    st.subheader("1. Pilih atau Unggah Citra Uji")
    input_options = ["Pilih Sampel Uji CIFAR-10", "Unggah Gambar Sendiri (Upload)"]
    input_method = st.radio("Metode Input:", input_options, index=1 if query_view == "upload" else 0, horizontal=True)

    selected_image = None
    image_source_label = ""

    if input_method == "Pilih Sampel Uji CIFAR-10":
        if samples:
            sample_keys = list(samples.keys())
            default_frog_idx = sample_keys.index(query_sample) if query_sample in sample_keys else (sample_keys.index("frog") if "frog" in sample_keys else 0)
            selected_sample_class = st.selectbox(
                "Pilih Sampel Kelas Asli CIFAR-10:",
                options=sample_keys,
                index=default_frog_idx
            )
            selected_image = samples[selected_sample_class]
            image_source_label = f"Sampel Asli Kelas: {selected_sample_class}"
        else:
            st.warning("Data sampel CIFAR-10 tidak dapat dimuat.")
    else:
        uploaded_file = st.file_uploader("Unggah berkas citra (JPG, JPEG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_source_label = f"Unggahan: {uploaded_file.name} ({selected_image.size[0]}×{selected_image.size[1]} px)"
            except Exception as e:
                st.error(f"Berkas citra tidak valid atau rusak: {e}")
                selected_image = None

    # Prediction & Visual Breakdown
    if selected_image is not None:
        st.markdown("---")
        st.subheader("2. Hasil Inferensi & Analisis Probabilitas")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"**Pratinjau Citra:** *{image_source_label}*")
            st.image(selected_image, use_container_width=True, caption=f"Tampilan Asli ({selected_image.size[0]}×{selected_image.size[1]} px)")

            # Resized 32x32 view
            img_32 = selected_image.resize((32, 32), Image.Resampling.BILINEAR)
            st.image(img_32, width=128, caption="Resolusi Input Model (32×32 px)")

        with col2:
            # Run Prediction
            top_class, top_prob, all_probs = predictor.predict(selected_image)

            # Prediction Card
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-title">Prediksi Kelas Model Terpilih</div>
                <div class="pred-class">{top_class.upper()}</div>
                <div class="pred-conf">Tingkat Keyakinan (Confidence): {top_prob*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown("**Distribusi Probabilitas Seluruh Kelas (Softmax Output):**")

            # Probability Dataframe & Chart
            prob_df = pd.DataFrame({
                'Kelas': list(all_probs.keys()),
                'Probabilitas': [p * 100 for p in all_probs.values()]
            }).sort_values(by='Probabilitas', ascending=True)

            chart = alt.Chart(prob_df).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('Probabilitas:Q', title='Probabilitas (%)', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Kelas:N', sort=None, title='Kelas CIFAR-10'),
                color=alt.condition(
                    alt.datum.Kelas == top_class,
                    alt.value('#1e3a8a'),
                    alt.value('#93c5fd')
                ),
                tooltip=[alt.Tooltip('Kelas:N'), alt.Tooltip('Probabilitas:Q', format='.2f')]
            ).properties(height=320)

            st.altair_chart(chart, use_container_width=True)

    else:
        st.info("Silakan pilih sampel citra atau unggah berkas untuk memulai inferensi.")

    # Footer
    st.markdown("---")
    st.caption("Proyek Deep Learning CIFAR-10 Image Classification with Deep CNNs | PyTorch & Streamlit | 2026")


if __name__ == '__main__':
    main()
