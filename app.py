from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('xray_model.h5')  # adjust path

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # adjust size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'xray' not in request.files:
        return "No file uploaded"

    file = request.files['xray']
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    img_array = preprocess(filepath)
    prediction = model.predict(img_array)[0][0]

    result = 'Positive (Pneumonia)' if prediction > 0.5 else 'Negative (Normal)'
    return render_template('index.html', prediction=result, image=filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
