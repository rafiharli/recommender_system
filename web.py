import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown

# Konfigurasi halaman
st.set_page_config(page_title="🎬 Movie Recommender System", layout="wide", page_icon="🎥")

# Fungsi download dari Google Drive
def download_from_gdrive(file_id, output):
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)

# File ID Google Drive
id_mappings = '1TZi8XmSfMY8LsTzwl84CKxk7FaC5UqHj'
svd_model_id = '17QcjpBCMjMiqkTPXI_yzVkRry6GgSfxn'
knn_model_id = '1OI6Umiz9n7INA38a1p4-TwbqwF6TYqny'
knn_sim_matrix_id = '1dSU0zz8874Kqzaq09WmjqmCTmOjVl7O_'
svd_sim_matrix_id = '1kgS98CKMPo7bmS2Gans2BrxgY7LNoGyH'

# Unduh file jika belum ada
download_from_gdrive(id_mappings, 'id_mappings.pkl')
download_from_gdrive(svd_model_id, 'svd_model.pkl')
download_from_gdrive(knn_model_id, 'knn_model.pkl')
download_from_gdrive(knn_sim_matrix_id, 'knn_sim_matrix.pkl')
download_from_gdrive(svd_sim_matrix_id, 'svd_sim_matrix.pkl')

# Load model dan data
with open("svd_model.pkl", "rb") as f:
    svd = pickle.load(f)
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)
with open("svd_sim_matrix.pkl", "rb") as f:
    svd_sim_matrix = pickle.load(f)
with open("knn_sim_matrix.pkl", "rb") as f:
    knn_sim_matrix = pickle.load(f)
with open("id_mappings.pkl", "rb") as f:
    inner_id_to_raw_id, raw_id_to_inner_id = pickle.load(f)

ratings_df = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv", encoding='latin1')
images_df = pd.read_csv("ml1m_images.csv")

movies_df = pd.merge(movies_df, images_df, on='movieId', how='left')

available_movie_ids = set(raw_id_to_inner_id.keys())
movies_df = movies_df[movies_df['movieId'].isin(available_movie_ids)]

# ========================= UI LAYOUT FUTURISTIC =========================
def start_layout():
    st.markdown("""
        <style>
        html, body, [data-testid="stApp"] {
            height: 100%;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }

        .fullscreen-wrapper {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        .content-wrapper {
            flex: 1;
            padding-bottom: 100px;
        }

        .custom-footer {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
            padding: 1.2rem 0;
            color: #ccc;
            font-size: 0.85rem;
            margin-top: auto;
            width: 100%;
            animation: fadeInUp 1s ease-in-out;
        }

        @keyframes fadeInUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .stSelectbox > div, .stNumberInput > div {
            background-color: #1e1e2f !important;
            color: white !important;
            border-radius: 10px;
        }

        a {
            color: #00d4ff;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        h1, h2, h3 {
            color: #f1f1f1;
        }

        .css-1v0mbdj {  /* streamlit default column */
            background-color: transparent !important;
        }
        </style>

        <div class="fullscreen-wrapper">
        <div class="content-wrapper">
    """, unsafe_allow_html=True)

def end_layout_with_footer():
    st.markdown("""
        </div>
        <div class="custom-footer">
            🚀 <strong>Movie Recommender by Rafi Harlianto</strong> | Hybrid AI (SVD + KNN) Engine<br>
            Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a> | © 2025 All Rights Reserved.
        </div>
        </div>
    """, unsafe_allow_html=True)

# ========================= SIDEBAR DAN NAVIGASI =========================
page = st.sidebar.selectbox("📂 Navigasi", ["Halaman Awal", "Rekomendasi Film"])
st.sidebar.markdown("---")
st.sidebar.markdown("👤 **Rafi Harlianto**\n\n📅 2025\n\n💡 Hybrid Recommender Engine")

# ========================= KONTEN UTAMA =========================
start_layout()

if page == "Halaman Awal":
    st.title("🎬 Selamat Datang di Sistem Rekomendasi Film")
    st.markdown("""
    Sistem ini menggunakan pendekatan **Weighted Hybrid** yang menggabungkan dua teknik canggih:
    - 🎯 **SVD (Singular Value Decomposition)** — berbasis *user preference*
    - 🧠 **KNN Item-based** — berbasis *kemiripan antar film*
    
    💡 Sistem ini mampu menampilkan rekomendasi **yang sesuai dengan selera pengguna** dengan visualisasi gambar poster yang menarik.
    """)

elif page == "Rekomendasi Film":
    st.title("🔎 Temukan Rekomendasi Film yang Mirip")

    selected_title = st.selectbox("📽️ Pilih judul film:", movies_df['title'].sort_values().unique())
    selected_movie = movies_df[movies_df['title'] == selected_title].iloc[0]
    selected_movie_id = int(selected_movie['movieId'])

    alpha = 0.8
    top_n = st.number_input("🎯 Jumlah Top-N Rekomendasi:", min_value=1, max_value=20, value=5, step=1)

    try:
        inner_i = raw_id_to_inner_id[selected_movie_id]
        sim_scores = []

        for j in range(svd_sim_matrix.shape[0]):
            if j == inner_i:
                continue
            sim_svd = svd_sim_matrix[inner_i][j]
            sim_knn = knn_sim_matrix[inner_i][j]
            sim = alpha * sim_svd + (1 - alpha) * sim_knn
            sim_scores.append((j, sim))

        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = sim_scores[:top_n]

        st.subheader(f"✨ Top {top_n} film mirip dengan **{selected_title}**")

        cols = st.columns(min(5, top_n))
        for idx, (inner, score) in enumerate(top_items):
            raw_id = int(inner_id_to_raw_id[inner])
            movie = movies_df[movies_df['movieId'] == raw_id].iloc[0]
            title = movie['title']
            img_url = movie['img_link'] if pd.notna(movie['img_link']) else "https://via.placeholder.com/150"

            pred_rating = score * 4 + 1
            rating_str = f"{pred_rating:.2f}".replace(".", ",")

            with cols[idx % len(cols)]:
                st.image(img_url, caption=f"**{title}**\n⭐ Prediksi: {rating_str}", use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mencari rekomendasi: {e}")

end_layout_with_footer()
