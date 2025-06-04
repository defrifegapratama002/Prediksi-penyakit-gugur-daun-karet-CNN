# api/index.py
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import sys
import tempfile
from PIL import Image
import io
import base64

# Setup paths untuk Vercel
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, '..', 'model')

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Konfigurasi upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load model - dengan error handling
model = None
try:
    MODEL_ARSITEKTUR = os.path.join(model_path, 'model_defrifega_1.json')
    MODEL_WEIGHTS = os.path.join(model_path, 'model_defrifega_weight.h5')
    
    if os.path.exists(MODEL_ARSITEKTUR) and os.path.exists(MODEL_WEIGHTS):
        with open(MODEL_ARSITEKTUR, 'r') as json_file:
            loaded_model_json = json_file.read()
        
        model = model_from_json(loaded_model_json)
        model.load_weights(MODEL_WEIGHTS)
        print('@@ Model loaded successfully')
    else:
        print('@@ Model files not found')
except Exception as e:
    print(f'@@ Error loading model: {str(e)}')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(img_data, model):
    """
    Prediksi menggunakan data image buffer (untuk Vercel serverless)
    """
    try:
        # Buat temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(img_data)
            tmp_path = tmp_file.name
        
        # Load dan preprocess image
        test_image = load_img(tmp_path, target_size=(128, 128))
        test_image = img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        # Prediksi
        result = model.predict(test_image)
        pred = np.argmax(result, axis=1)[0]
        confidence = float(np.max(result))
        
        # Cleanup temporary file
        os.unlink(tmp_path)
        
        # Mapping hasil prediksi
        predictions = {
            0: ('Daun terkena bercak tepung putih Oidium', 'bercak putih.html'),
            1: ('Daun terkena jamur Colltotrichum gloeosporioides', 'Colltotrichum.html'),
            2: ('Daun bercak coklat', 'bercak coklat.html'),
            3: ('Daun sehat', 'sehat.html')
        }
        
        pred_text, output_page = predictions.get(pred, ('Tidak dapat mengklasifikasi', 'error.html'))
        
        return pred_text, output_page, confidence
        
    except Exception as e:
        print(f'@@ Error in prediction: {str(e)}')
        return 'Error dalam prediksi', 'error.html', 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/modelling')
def modelling():
    return render_template('modelling.html')

@app.route("/prediksi", methods=['GET', 'POST'])
def home():
    return render_template('pendeteksi.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if model is loaded
        if model is None:
            return render_template('error.html', 
                                 error_message='Model tidak dapat dimuat. Silakan coba lagi nanti.')
        
        # Check if file is uploaded
        if 'image' not in request.files:
            return render_template('pendeteksi.html', 
                                 error='Tidak ada file yang dipilih')
        
        file = request.files['image']
        
        if file.filename == '':
            return render_template('pendeteksi.html', 
                                 error='Tidak ada file yang dipilih')
        
        if file and allowed_file(file.filename):
            try:
                # Baca file ke memory (untuk serverless)
                file_data = file.read()
                filename = secure_filename(file.filename)
                
                print(f"@@ Processing file: {filename}")
                
                # Prediksi
                pred_text, output_page, confidence = model_predict(file_data, model)
                
                # Convert image to base64 untuk display
                img_base64 = base64.b64encode(file_data).decode('utf-8')
                img_data_url = f"data:image/jpeg;base64,{img_base64}"
                
                return render_template(output_page, 
                                     pred_output=pred_text,
                                     user_image=img_data_url,
                                     confidence=f"{confidence:.2%}")
                
            except Exception as e:
                print(f"@@ Error processing image: {str(e)}")
                return render_template('error.html', 
                                     error_message=f'Error memproses gambar: {str(e)}')
        else:
            return render_template('pendeteksi.html', 
                                 error='Format file tidak didukung. Gunakan PNG, JPG, atau JPEG')
    
    return render_template('pendeteksi.html')

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return render_template('error.html', 
                         error_message='File terlalu besar. Maksimal 16MB'), 413

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', 
                         error_message='Terjadi kesalahan internal server'), 500

if __name__ == '__main__':
    app.run(debug=True)