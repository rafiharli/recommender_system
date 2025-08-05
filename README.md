# 🎬 Sistem Rekomendasi Film

Sistem rekomendasi film berbasis machine learning menggunakan dataset **MovieLens 100K** dan **MovieLens 1M**. Sistem ini dikembangkan menggunakan model:

- **SVD (Singular Value Decomposition)**
- **KNN (K-Nearest Neighbors)(Item-Based dengan Cosine Similarity)**

Model dibangun dengan bantuan pustaka [Surprise](http://surpriselib.com/), yang dirancang khusus untuk sistem rekomendasi.

---

## 📁 Dataset

Sistem ini menggunakan dua versi dari dataset MovieLens:

- [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

Untuk pengembangan silahkan download zip file ini atau dengan clone menggunakan CMD atau powershell:

Download .ZIP:

**https://github.com/rafiharli/recommender_system/archive/refs/heads/main.zip**

Clone dengan CMD atau PowerShell dengan perintah:

**git clone https://github.com/rafiharli/recommender_system.git**

## 🛠️ Instalasi

### Persyaratan

- gdown>=5.0.0
- streamlit>=1.45.0
- pandas>=2.2.3
- numpy>=1.26.4
- scikit-learn>=1.5.2
- scikit-surprise>=1.1.4

### Instalasi Dependency

Untuk menginstall dependency sesuai dengan persyaratan sistem gunakan "pip install"

contoh: **pip install streamlit**
