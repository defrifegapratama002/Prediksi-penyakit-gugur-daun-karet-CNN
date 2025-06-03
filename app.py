from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os
import sys
import glob
import re
import h5py
import PIL
from PIL import Image

app = Flask(__name__)

MODEL_ARSITEKTUR =  'model/model_defrifega_1.json'   
MODEL_WEIGHTS = 'model/model_defrifega_weight.h5'

json_file = open(MODEL_ARSITEKTUR)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(MODEL_WEIGHTS)
print('@@ Model loaded. Check http://127.0.0.1:5000/')

@app.route ('/')
def index():
    return render_template('index.html')
@app.route ('/about')
def about():
    return render_template('about.html')
@app.route ('/modelling')
def modelling():
    return render_template('modelling.html')

def model_predict(img_path, model):
  test_image = load_img(img_path, target_size = (128, 128)) # load image 
  print("@@ Got Image for prediction")
  test_image = img_to_array(test_image)/255 # normalisasi dan ubah ke array
  test_image = np.expand_dims(test_image, axis = 0) 
  result = model.predict(test_image) # prediksi 
  pred = np.argmax(result, axis=1) # ambil indexnya
         
  if   pred == 0: 
      return 'Daun terkena bercak tepung putih Oidium', 'bercak putih.html'
  elif pred == 1:
      return 'Daun terkena jamur Colltotrichum gloeosporioides', 'Colltotrichum.html'   
  elif pred == 2:
      return 'Daun bercak coklat', 'bercak coklat.html'  
  else:
      return "Daun sehat", 'sehat.html'
  
  #saya belum membuat untuk else dari ke 4 objek ini
    
@app.route("/prediksi", methods=['GET', 'POST'])
def home():
        return render_template('pendeteksi.html')
     
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] 
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = model_predict(file_path, model)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    

if __name__ == "__main__":
    app.run(threaded=False,) 
    
    