#Create virtual environment:
#  python3 -m venv env
#Activate virtual environment:
#  source env/bin/activate

from flask import Flask, render_template,request,abort
import pandas as pd
import numpy as np
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras import layers
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'
model = tf.keras.models.load_model('CNN_COVID_ClassifierV1.h5') #Won't be uploaded.


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predictor.html')
	
@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
   if request.method == 'POST':
       img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_COLOR)
       try:
           image = cv2.resize(img, (128, 128))
           image = [image]
           image = np.array(image)
           prediction = int(np.argmax(model.predict(image)[0]))

           if prediction ==0:
               return render_template('prediction_0.html')
           if prediction ==1:
               return render_template('prediction_1.html')
           if prediction ==2:
               return render_template('prediction_2.html')
       except:
           return abort(404)
	

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080,debug=True)