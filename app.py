# app.py (Versi 2.0 - Multi-Halaman & Grafik)

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Muat Model & Nama Kelas (Sama seperti sebelumnya) ---
model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
print(f"Model dari {model_path} berhasil dimuat.")

# --- Fungsi Preprocessing (Sama seperti sebelumnya) ---
def preprocess_image(image_file):
    img = Image.open(image_file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    normalized_img_array = (img_array.astype(np.float32) / 255.0)
    return normalized_img_array

# --- RUTE (URL) BARU UNTUK WEBSITE ---

# Rute 1: Halaman Utama (Home Page)
@app.route('/')
def home():
    # Akan menampilkan file 'home.html'
    return render_template('home.html')

# Rute 2: Halaman Klasifikasi
@app.route('/classify')
def classify():
    # Akan menampilkan file 'classify.html' (yang dulu 'index.html')
    return render_template('classify.html')


# Rute 4: Endpoint Prediksi (Diperbarui)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang dikirim'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    try:
        processed_image = preprocess_image(file)
        prediction = model.predict(processed_image)
        
        # --- PERUBAHAN PENTING ADA DI SINI ---
        
        # 1. Dapatkan kelas prediksi utama (seperti sebelumnya)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(prediction[0])) * 100

        # 2. Dapatkan SEMUA probabilitas untuk grafik (BARU)
        # Ubah numpy array menjadi list Python biasa
        probabilities = prediction[0].tolist() 
        # Bulatkan probabilitas untuk dikirim ke frontend
        probabilities_percent = [round(p * 100, 2) for p in probabilities]

        return jsonify({
            'predicted_class': predicted_class_name,
            'confidence': f"{confidence:.2f}%",
            'probabilities': probabilities_percent, # Data baru untuk grafik
            'class_names': class_names # Kirim nama kelas untuk label grafik
        })

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat memproses: {e}'}), 500

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True)