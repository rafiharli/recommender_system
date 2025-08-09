import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os

st.set_page_config(page_title="Movie Recommender System", layout="wide")

# ================== KONFIGURASI ID GDRIVE ==================
FILE_IDS = {
    "id_mappings": "1eqH1JUM8Thw4F1sYU9qJSdqb5oNNtstm",
    "svd_model": "1qIs2bJttxOlk6TA4VNP-geeKAH21Y45t",
    "knn_model": "1l2eRmLZpei09EDDbqMDUlJpdIJp0ukNb",
    "knn_matrix": "1VJ_cDKBcqyzhjc48oC3zBFebLfBgPSz0",
    "svd_matrix": "1ZbZRYoqYH1BWBPAjiAK1Jd6dpOR2ni87"
}

NEW_FILE_IDS = {
    "new_id_mappings": "1P1ixF4R__DeEDIy9EZilDM8BvD9ml4i5",
    "new_svd_model": "1jzEDNrvhQ52cAXm5DMIVdqdHmRy9kU-_",
    "new_knn_model": "1rEMLaVOTwY-dqVgOSV8az5_7u26fDxUA",
    "new_knn_matrix": "1Tjhvxb5kzXLXbnAQ_rowhk4vuff1lxok",
    "new_svd_matrix": "19YoWS6aNUk0NC0cPwLvX1iGDZ9YM77Nd"
}

# ================== DOWNLOAD FILE JIKA BELUM ADA ==================
def download_from_gdrive(file_id, output):
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)

# ================== LOAD DAN CACHE MODEL ==================
@st.cache_resource
def load_models():
    download_from_gdrive(FILE_IDS["id_mappings"], "id_mappings.pkl")
    download_from_gdrive(FILE_IDS["svd_model"], "svd_model.pkl")
    download_from_gdrive(FILE_IDS["knn_model"], "knn_model.pkl")
    download_from_gdrive(FILE_IDS["knn_matrix"], "knn_matrix.pkl")
    download_from_gdrive(FILE_IDS["svd_matrix"], "svd_matrix.pkl")

    with open("svd_model.pkl", "rb") as f:
        svd = pickle.load(f)
    with open("knn_model.pkl", "rb") as f:
        knn = pickle.load(f)
    with open("svd_matrix.pkl", "rb") as f:
        svd_matrix = pickle.load(f)
    with open("knn_matrix.pkl", "rb") as f:
        knn_matrix = pickle.load(f)
    with open("id_mappings.pkl", "rb") as f:
        inner_id_to_raw_id, raw_id_to_inner_id = pickle.load(f)

    return svd, knn, svd_matrix, knn_matrix, inner_id_to_raw_id, raw_id_to_inner_id

@st.cache_resource
def load_new_models():
    download_from_gdrive(NEW_FILE_IDS["new_id_mappings"], "new_id_mappings.pkl")
    download_from_gdrive(NEW_FILE_IDS["new_svd_model"], "new_svd_model.pkl")
    download_from_gdrive(NEW_FILE_IDS["new_knn_model"], "new_knn_model.pkl")
    download_from_gdrive(NEW_FILE_IDS["new_knn_matrix"], "new_knn_matrix.pkl")
    download_from_gdrive(NEW_FILE_IDS["new_svd_matrix"], "new_svd_matrix.pkl")

    with open("new_svd_model.pkl", "rb") as f:
        svd_new = pickle.load(f)
    with open("new_knn_model.pkl", "rb") as f:
        knn_new = pickle.load(f)
    with open("new_svd_matrix.pkl", "rb") as f:
        svd_matrix_new = pickle.load(f)
    with open("new_knn_matrix.pkl", "rb") as f:
        knn_matrix_new = pickle.load(f)
    with open("new_id_mappings.pkl", "rb") as f:
        inner_id_to_raw_id_new, raw_id_to_inner_id_new = pickle.load(f)

    return svd_new, knn_new, svd_matrix_new, knn_matrix_new, inner_id_to_raw_id_new, raw_id_to_inner_id_new

# ================== LOAD DATASET CSV ==================
@st.cache_data
def load_datasets():
    ratings_url = 'https://raw.githubusercontent.com/rafiharli/recommender_system/main/ratings.csv'
    movies_url = 'https://raw.githubusercontent.com/rafiharli/recommender_system/main/movies.csv'
    images_url = 'https://raw.githubusercontent.com/rafiharli/recommender_system/main/ml1m_images.csv'

    ratings_df = pd.read_csv(ratings_url)
    movies_df = pd.read_csv(movies_url, encoding='latin1')
    images_df = pd.read_csv(images_url)

    return ratings_df, movies_df, images_df

@st.cache_data
def load_new_datasets():
    ratings_df_new = pd.read_csv("ratings_2020.csv")
    movies_df_new = pd.read_csv("new_movies.csv", encoding='latin1')
    return ratings_df_new, movies_df_new

# ================== LOAD SEMUA ==================
svd, knn, svd_matrix, knn_matrix, inner_id_to_raw_id, raw_id_to_inner_id = load_models()
ratings_df, movies_df, images_df = load_datasets()

movies_df = pd.merge(movies_df, images_df, on='movieId', how='left')
available_movie_ids = set(raw_id_to_inner_id.keys())
movies_df = movies_df[movies_df['movieId'].isin(available_movie_ids)]

# ================== NAVIGASI SIDEBAR ==================
if 'page' not in st.session_state:
    st.session_state.page = "Halaman Awal"

st.sidebar.title("Navigasi")
if st.sidebar.button("🏠 Halaman Awal"):
    st.session_state.page = "Halaman Awal"
if st.sidebar.button("🎯 Rekomendasi Film"):
    st.balloons()
    st.session_state.page = "Rekomendasi Film"
if st.sidebar.button("🆕 Rekomendasi Film (Dataset Baru)"):
    st.session_state.page = "Rekomendasi Film (Dataset Baru)"

page = st.session_state.page

# ================== HALAMAN AWAL ==================
if page == "Halaman Awal":
    st.title("🎬 Sistem Rekomendasi Film")
    st.markdown("""
    Selamat datang di sistem rekomendasi film berbasis **Weighted Hybrid (SVD + KNN)**!

    🔍 **Rekomendasi Film** dengan **Prediksi Film** yang sesuai dengan **Preferensi Pengguna**.  
    ⚖️ Dapat menyesuaikan Top-N jumlah rekomendasi.  
    🎥 Ditampilkan dengan poster film.
    """)

# ================== HALAMAN REKOMENDASI LAMA ==================
elif page == "Rekomendasi Film (Film <= 2000)":
    st.title("🎯 Cari Rekomendasi Film")
    alpha = 0.8
    selected_title = st.selectbox("Ketik atau pilih judul film:", movies_df['title'])
    selected_movie = movies_df[movies_df['title'] == selected_title].iloc[0]
    selected_movie_id = int(selected_movie['movieId'])
    top_n = st.number_input("Top-N rekomendasi", min_value=1, max_value=20, value=10, step=1)

    try:
        with st.spinner("🔎 Mencari film yang mirip..."):
            inner_i = raw_id_to_inner_id[selected_movie_id]
            pred_scores = []

            for j in range(svd_matrix.shape[0]):
                if j == inner_i:
                    continue
                pred_svd = svd_matrix[inner_i][j]
                pred_knn = knn_matrix[inner_i][j]
                pred_weighted = alpha * pred_svd + (1 - alpha) * pred_knn
                pred_scores.append((j, pred_weighted))

            pred_scores.sort(key=lambda x: x[1], reverse=True)
            top_items = pred_scores[:top_n]

            st.subheader(f'Top {top_n} Rekomendasi Film "*{selected_title}*"')
            cols = st.columns(min(5, top_n))

            for idx, (inner, score) in enumerate(top_items):
                raw_id = int(inner_id_to_raw_id[inner])
                movie = movies_df[movies_df['movieId'] == raw_id].iloc[0]
                title = movie['title']
                img_url = movie['img_link'] if pd.notna(movie['img_link']) else "https://via.placeholder.com/150?text=No+Image"
                pred_rating = score * 4 + 1
                rating_str = f"⭐ {pred_rating:.2f}/5"

                with cols[idx % len(cols)]:
                    st.image(img_url, caption=f"{title}\n\n{rating_str}", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat mencari rekomendasi: {e}")

# ================== HALAMAN REKOMENDASI BARU ==================
elif page == "Rekomendasi Film (Film >= 2020)":
    st.title("🆕 Rekomendasi Film (Dataset 2020+)")
    alpha = 0.8

    svd_new, knn_new, svd_matrix_new, knn_matrix_new, inner_id_to_raw_id_new, raw_id_to_inner_id_new = load_new_models()
    ratings_df_new, movies_df_new = load_new_datasets()

    available_movie_ids_new = set(raw_id_to_inner_id_new.keys())
    movies_df_new = movies_df_new[movies_df_new['movieId'].isin(available_movie_ids_new)]

    selected_title_new = st.selectbox("Pilih film (dataset baru):", movies_df_new['title'])
    selected_movie_new = movies_df_new[movies_df_new['title'] == selected_title_new].iloc[0]
    selected_movie_id_new = int(selected_movie_new['movieId'])
    top_n_new = st.number_input("Top-N rekomendasi", min_value=1, max_value=20, value=10, step=1, key="topn_new")

    try:
        with st.spinner("🔎 Mencari rekomendasi (dataset baru)..."):
            inner_i_new = raw_id_to_inner_id_new[selected_movie_id_new]
            pred_scores_new = []

            for j in range(svd_matrix_new.shape[0]):
                if j == inner_i_new:
                    continue
                pred_svd_new = svd_matrix_new[inner_i_new][j]
                pred_knn_new = knn_matrix_new[inner_i_new][j]
                pred_weighted_new = alpha * pred_svd_new + (1 - alpha) * pred_knn_new
                pred_scores_new.append((j, pred_weighted_new))

            pred_scores_new.sort(key=lambda x: x[1], reverse=True)
            top_items_new = pred_scores_new[:top_n_new]

            st.subheader(f"Top {top_n_new} rekomendasi untuk '{selected_title_new}'")
            for idx, (inner, score) in enumerate(top_items_new, start=1):
                raw_id_new = int(inner_id_to_raw_id_new[inner])
                movie_new = movies_df_new[movies_df_new['movieId'] == raw_id_new].iloc[0]
                pred_rating_new = score * 4 + 1
                st.write(f"{idx}. {movie_new['title']} — ⭐ {pred_rating_new:.2f}/5")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat mencari rekomendasi: {e}")

# ================== FOOTER ==================
def render_footer():
    st.markdown("""<hr>""", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.85rem; color: gray; margin-top: 20px;'>
            © 2025 <strong>Rafi Harlianto</strong> • Dibuat dengan ❤️ menggunakan <a href='https://streamlit.io' target='_blank'>Streamlit</a>
        </div>
        """,
        unsafe_allow_html=True
    )

render_footer()
