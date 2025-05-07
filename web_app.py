# app.py
import streamlit as st
import pandas as pd
from surprise import SVD, KNNBasic, Dataset, Reader

st.set_page_config(page_title="Sistem Rekomendasi Film")

# === HANYA DIJALANKAN SEKALI ===
if 'model_svd' not in st.session_state or 'model_knn' not in st.session_state:
    st.info("Meload dataset dan melatih model...")

    # Load ratings dan movies
    reader = Reader(line_format='user item rating timestamp', sep='::')
    data = Dataset.load_from_file("ratings.dat", reader=reader)
    trainset = data.build_full_trainset()

    # Train models
    model_svd = SVD()
    model_svd.fit(trainset)
    st.session_state.model_svd = model_svd

    sim_options = {'name': 'cosine', 'user_based': False}
    model_knn = KNNBasic(sim_options=sim_options)
    model_knn.fit(trainset)
    st.session_state.model_knn = model_knn

    # Simpan juga movies dan ratings agar tidak di-load tiap rerun
    movies = pd.read_csv("movies.dat",
                         sep="::", engine="python",
                         names=["movieId", "title", "genres"], encoding="latin-1")
    ratings = pd.read_csv("ratings.dat",
                          sep="::", engine="python",
                          names=["userId","movieId", "rating", "timestamp"])
    st.session_state.movies = movies
    st.session_state.ratings = ratings

else:
    model_svd = st.session_state.model_svd
    model_knn = st.session_state.model_knn
    movies = st.session_state.movies
    ratings = st.session_state.ratings

# Session states
if 'login' not in st.session_state:
    st.session_state.login = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'rated_movies' not in st.session_state:
    st.session_state.rated_movies = set()
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to get weighted recommendations (SVD + KNN)
def recommend_movies(user_id, top_n=5, svd_weight=0.7, knn_weight=0.3):
    user_seen = set(ratings[ratings['userId'] == user_id]['movieId'])
    all_movies = set(movies['movieId'])
    unseen = list(all_movies - user_seen)

    predictions = []
    for movie_id in unseen:
        svd_score = model_svd.predict(user_id, movie_id).est
        knn_score = model_knn.predict(user_id, movie_id).est
        hybrid_score = (svd_weight * svd_score) + (knn_weight * knn_score)
        predictions.append((movie_id, hybrid_score))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:top_n]

    results = []
    for movie_id, score in top_movies:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        results.append((title, f"(Skor prediksi: {score:.2f})"))
    return results


# Navigation logic
def show_navigation():
    st.sidebar.title("Navigasi")
    if st.session_state.login:
        if st.sidebar.button("Rekomendasi Film", key="nav_rekom"):
            st.session_state.page = 'rekomendasi'
        if st.sidebar.button("Logout", key="nav_logout"):
            logout()
    else:
        if st.sidebar.button("Login", key="nav_login"):
            st.session_state.page = 'login'

def logout():
    st.session_state.login = False
    st.session_state.user_id = None
    st.session_state.rated_movies = set()
    st.session_state.page = 'home'
    st.rerun()

# Pages
show_navigation()

st.title("Sistem Rekomendasi Film")
 
# LOGIN
if not st.session_state.login and st.session_state.page == "login":
    st.subheader("Login")
    user_id_input = st.text_input("Masukkan User ID [1-6040]")
    
    if st.button("Login", key="login_submit"):
        # Cek apakah input valid (hanya angka)
        if user_id_input.isdigit():
            user_id = int(user_id_input)
            if user_id in ratings['userId'].unique():
                st.session_state.login = True
                st.session_state.user_id = user_id
                st.session_state.rated_movies = set(ratings[ratings['userId'] == user_id]['movieId'])
                st.success("Login berhasil!")
                st.session_state.page = 'rekomendasi'
                st.rerun()
            else:
                st.error("User ID tidak ditemukan.")
        else:
            st.error("Masukkan User ID yang valid (angka).")

# REKOMENDASI
elif st.session_state.login and st.session_state.page == "rekomendasi":
    st.subheader(f"Selamat datang, User ID: {st.session_state.user_id}")
    st.subheader("Rekomendasi Film Untuk Anda")
    top_n = st.slider("Jumlah rekomendasi yang diinginkan:", 1, 20, 5)
    if len(st.session_state.rated_movies) >= 5:
        recs = recommend_movies(st.session_state.user_id, top_n=top_n)
        for idx, (title, reason) in enumerate(recs, 1):  # Mulai nomor dari 1
            st.write(f"{idx}. **{title}** - {reason}")
    else:
        st.warning("Rating Pengguna terhadap Film Kurang.")

# HOME PAGE DEFAULT
elif st.session_state.page == 'home':
    st.write("Selamat datang di sistem rekomendasi film dengan pendekatan _collaborative filtering_ menggunakan metode weighted hybrid (SVD-KNN). Silahkan Login dengan UserID")