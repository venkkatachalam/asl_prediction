
# IP Address : PORT --------- 60k+ ports 
# App1 --- IP ADD:Port1 10.1.0.254:5000
# App2 --- IP ADD:Port2 10.1.0.254:5001
# 10.1.0.254:5000

from flask import Flask, jsonify, request, send_file
import numpy as np
from PIL import Image
import io
import os

import keras

app = Flask(__name__) # __Name__ = __Main__ ---------> app ------------> The locality of my app /templates /static

print("Load model")
model = keras.models.load_model('ASL.h5')

input_shape = model.input_shape
IMG_SIZE = input_shape[1]
# Load the model

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
# Define our server and the API Paths.
@app.route('/')
def home():
    return send_file('./UX/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No Image Provided'}), 400 # BAD REQ

    # Process the image
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = img.convert('RGB')
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0) # Batch * Height * width * channels(RGB)

    prediction = model.predict(img_array, verbose=0)
    predicted_class = classes[np.argmax(prediction)] #[0.01, 0.02....0.01] ---> 1 ()
    confidence = float(np.max(prediction))

    # Predict

    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence
    })
## Load the model
## Help with Prediction

if __name__ == '__main__':
    app.run(port=5000)