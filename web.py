import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os


# Konfigurasi halaman
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# --- FUNGSI UNDUH GOOGLE DRIVE ---
def download_from_gdrive(file_id, output):
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)

# --- FILE ID DARI GOOGLE DRIVE ---
id_mappings = '1eqH1JUM8Thw4F1sYU9qJSdqb5oNNtstm'
svd_model_id = '1qIs2bJttxOlk6TA4VNP-geeKAH21Y45t'
knn_model_id = '1l2eRmLZpei09EDDbqMDUlJpdIJp0ukNb'
knn_matrix_id = '1VJ_cDKBcqyzhjc48oC3zBFebLfBgPSz0'
svd_matrix_id = '1ZbZRYoqYH1BWBPAjiAK1Jd6dpOR2ni87'

# Unduh jika belum ada
download_from_gdrive(id_mappings, 'id_mappings.pkl')
download_from_gdrive(svd_model_id, 'svd_model.pkl')
download_from_gdrive(knn_model_id, 'knn_model.pkl')
download_from_gdrive(knn_matrix_id, 'knn_matrix.pkl')
download_from_gdrive(svd_matrix_id, 'svd_matrix.pkl')

# Load model dan data
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

ratings_df = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv", encoding='latin1')
images_df = pd.read_csv("ml1m_images.csv")

# Gabungkan data judul dan gambar
movies_df = pd.merge(movies_df, images_df, on='movieId', how='left')
available_movie_ids = set(raw_id_to_inner_id.keys())
movies_df = movies_df[movies_df['movieId'].isin(available_movie_ids)]

# Navigasi manual
if 'page' not in st.session_state:
    st.session_state.page = "Halaman Awal"

st.sidebar.title("Navigasi")
if st.sidebar.button("🏠 Halaman Awal"):
    st.session_state.page = "Halaman Awal"
if st.sidebar.button("🎯 Rekomendasi Film"):
    st.balloons()
    st.session_state.page = "Rekomendasi Film"

page = st.session_state.page

# Halaman Awal
if page == "Halaman Awal":
    st.title("🎬 Sistem Rekomendasi Film")
    st.markdown("""
    Selamat datang di sistem rekomendasi film berbasis **Weighted Hybrid (SVD + KNN)**!

    🔍 **Rekomendasi Film** dengan **Prediksi Film** yang sesuai dengan **Preferensi Pengguna**.  
    ⚖️ Dapat menyesuaikan Top-N jumlah rekomendasi.  
    🎥 Ditampilkan dengan poster film.
    """)

# Halaman Rekomendasi
elif page == "Rekomendasi Film":
    st.title("🎯 Cari Rekomendasi Film")
    alpha = 0.9
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

# Footer
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
