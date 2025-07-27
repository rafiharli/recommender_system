import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown

# Konfigurasi halaman
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# --- FUNGSI UNDUH GOOGLE DRIVE ---
def download_from_gdrive(file_id, output):
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)

# --- FILE ID DARI GOOGLE DRIVE ---
id_mappings = '1TZi8XmSfMY8LsTzwl84CKxk7FaC5UqHj'
svd_model_id = '17QcjpBCMjMiqkTPXI_yzVkRry6GgSfxn'
knn_model_id = '1OI6Umiz9n7INA38a1p4-TwbqwF6TYqny'
knn_sim_matrix_id = '1dSU0zz8874Kqzaq09WmjqmCTmOjVl7O_'
svd_sim_matrix_id = '1kgS98CKMPo7bmS2Gans2BrxgY7LNoGyH'

# Unduh jika belum ada
download_from_gdrive(id_mappings, 'knn_model.pkl')
download_from_gdrive(svd_model_id, 'knn_model.pkl')
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

# Gabungkan data judul dan gambar
movies_df = pd.merge(movies_df, images_df, on='movieId', how='left')

# 🔒 Filter hanya film yang tersedia dalam pelatihan
available_movie_ids = set(raw_id_to_inner_id.keys())
movies_df = movies_df[movies_df['movieId'].isin(available_movie_ids)]

# Sidebar navigasi
page = st.sidebar.selectbox("Navigasi", ["Halaman Awal", "Rekomendasi Film"])

# 🏠 Halaman Awal
if page == "Halaman Awal":
    st.title("🎬 Sistem Rekomendasi Film")
    st.markdown("""
    Selamat datang di sistem rekomendasi film berbasis **Weighted Hybrid (SVD + KNN)**!

    🔍 **Rekomendasi Film** dengan **Prediksi Film** yang sesuai dengan **Preferensi Pengguna**.  
    ⚖️ Dapat menyesuaikan Top-N jumlah rekomendasi.  
    🎥 Ditampilkan dengan poster film.
    """)

# 🎯 Halaman Rekomendasi
elif page == "Rekomendasi Film":
    st.title("🎯 Cari Rekomendasi Film")

    selected_title = st.selectbox("Ketik atau pilih judul film:", movies_df['title'].sort_values().unique())
    selected_movie = movies_df[movies_df['title'] == selected_title].iloc[0]
    selected_movie_id = int(selected_movie['movieId'])

    alpha = 0.9
    top_n = st.number_input("Top-N rekomendasi", min_value=1, max_value=20, value=5, step=1)

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

        st.subheader(f"Top {top_n} rekomendasi mirip dengan '{selected_title}'")
        cols = st.columns(min(5, top_n))

        for idx, (inner, score) in enumerate(top_items):
            raw_id = int(inner_id_to_raw_id[inner])
            movie = movies_df[movies_df['movieId'] == raw_id].iloc[0]
            title = movie['title']
            img_url = movie['img_link'] if pd.notna(movie['img_link']) else "https://via.placeholder.com/150"

            pred_rating = score * 4 + 1
            rating_str = f"{pred_rating:.3f}".replace(".", ",")

            with cols[idx % len(cols)]:
                st.image(img_url, caption=f"{title}\n\nPrediksi Rating: {rating_str}", use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mencari rekomendasi: {e}")
