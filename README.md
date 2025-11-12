# Sistem Deteksi Tumor Otak

Proyek ini adalah aplikasi web yang dibangun dengan Flask untuk mengklasifikasi citra MRI (Magnetic Resonance Imaging) otak. Aplikasi ini menggunakan model Convolutional Neural Network (CNN) yang dibangun dengan arsitektur **MobileNetV2** untuk memprediksi apakah sebuah gambar MRI menunjukkan tumor dan mengidentifikasi jenisnya.

## 🧠 Fitur Utama

- **Klasifikasi Multi-Kelas:** Model dapat membedakan antara 4 kategori:
  1.  Glioma Tumor
  2.  Meningioma Tumor
  3.  No Tumor (Tidak Ada Tumor)
  4.  Pituitary Tumor
- **Antarmuka Web:** Aplikasi web sederhana dengan dua halaman utama:
  - `/` (Beranda): Halaman selamat datang yang menjelaskan proyek.
  - `/classify`: Halaman untuk mengunggah gambar MRI dan melihat hasil klasifikasi.
- **API Prediksi:** Endpoint `/predict` (via POST) yang menerima file gambar dan mengembalikan hasil prediksi dalam format JSON.
- **Visualisasi Hasil:** Hasil prediksi tidak hanya menampilkan kelas dengan probabilitas tertinggi, tetapi juga menyertakan data untuk grafik batang yang menunjukkan skor probabilitas untuk keempat kelas.

## 🛠️ Teknologi yang Digunakan

- **Backend:** Python, Flask
- **Machine Learning / Deep Learning:** TensorFlow 2.x, Keras
- **Pemrosesan Data & Gambar:** NumPy, Pillow (PIL)
- **Lingkungan Pengembangan:** Jupyter Notebook (untuk pelatihan model)
- **Frontend:** HTML, CSS (dan JavaScript untuk visualisasi data di halaman `classify.html`)

## 📄 Struktur Proyek

tumor_otak/
│
├── app.py # File utama aplikasi Flask (Backend)
├── TrainingModel.ipynb # Notebook Jupyter untuk melatih model
├── model.h5 # Model Keras (.h5) yang sudah dilatih
│
└── templates/
├── layout.html # Template dasar (tidak disertakan, tapi direferensikan)
├── home.html # Halaman beranda
└── classify.html # Halaman untuk unggah & klasifikasi

## 🤖 Detail Model (TrainingModel.ipynb)

Model `model.h5` dilatih menggunakan teknik _Transfer Learning_.

- **Model Dasar:** **MobileNetV2** (dilatih pada dataset 'imagenet'), dengan lapisan atas (fully connected) dibuang (`include_top=False`).
- **Pembekuan (Freezing):** Seluruh lapisan konvolusi dari MobileNetV2 dibekukan (`base_model.trainable = False`) agar bobotnya tidak berubah saat pelatihan.
- **Classifier (Kepala Model):** Lapisan kustom ditambahkan di atas model dasar:
  1.  `GlobalAveragePooling2D()`: Meratakan output fitur.
  2.  `Dense(1024, activation='relu')`: Lapisan tersembunyi (hidden layer).
  3.  `Dropout(0.2)`: Untuk mengurangi overfitting.
  4.  `Dense(4, activation='softmax')`: Lapisan output untuk 4 kelas.
- **Dataset:**
  - Dataset: Brain Tumor MRI Dataset (dari Kaggle).
  - Ukuran: 2870 gambar training, 394 gambar testing.
  - Ukuran Gambar: Diubah menjadi `(224, 224)` piksel.
- **Augmentasi Data (Training):**
  - Normalisasi (Rescale 1./255)
  - Rotasi, Pergeseran (width/height), Shear, Zoom, Flip Horizontal.
- **Kompilasi Model:**
  - **Optimizer:** `Adam` (learning rate = 0.0001)
  - **Loss:** `categorical_crossentropy` (karena ini masalah klasifikasi multi-kelas)
- **Hasil Pelatihan:**
  - Model dilatih selama **70 Epochs**.
  - Akurasi akhir pada data tes (evaluasi): **73.86%**.

## 🚀 Instalasi dan Cara Menjalankan

Untuk menjalankan aplikasi web ini di komputer lokal Anda:

1.  **Clone atau Unduh Proyek:**
    Pastikan semua file (`app.py`, `model.h5`, `TrainingModel.ipynb`, dan folder `templates`) berada dalam satu direktori.

2.  **Buat Virtual Environment** (Direkomendasikan):

    ```bash
    python -m venv venv
    ```

    - Di Windows: `venv\Scripts\activate`
    - Di macOS/Linux: `source venv/bin/activate`

3.  **Install Dependensi:**
    Anda memerlukan library Python yang digunakan dalam file `app.py` dan `TrainingModel.ipynb`.

    ```bash
    pip install flask tensorflow numpy pillow
    ```

    _(Jika Anda ingin menjalankan notebook, tambahkan `matplotlib` dan `jupyter`)_

4.  **Jalankan Aplikasi Flask:**
    Buka terminal di direktori proyek dan jalankan:

    ```bash
    python app.py
    ```

5.  **Buka di Browser:**
    Aplikasi akan berjalan dalam mode debug di:
    `http://127.0.0.1:5000`
