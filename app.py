import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
print(f"Model dari {model_path} berhasil dimuat.")

# --- Fungsi Preprocessing  ---
def preprocess_image(image_file):
    img = Image.open(image_file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    normalized_img_array = (img_array.astype(np.float32) / 255.0)
    return normalized_img_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

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
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(prediction[0])) * 100

        probabilities = prediction[0].tolist() 
        probabilities_percent = [round(p * 100, 2) for p in probabilities]
        return jsonify({
            'predicted_class': predicted_class_name,
            'confidence': f"{confidence:.2f}%",
            'probabilities': probabilities_percent, 
            'class_names': class_names 
        })
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat memproses: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)