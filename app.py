
# IP Address : PORT --------- 60k+ ports 
# App1 --- IP ADD:Port1 10.1.0.254:5000
# App2 --- IP ADD:Port2 10.1.0.254:5001
# 10.1.0.254:5000

from flask import Flask, jsonify, request, send_file
import numpy as np
from PIL import Image
import io
import os
import tflite_runtime.interpreter as tflite

app = Flask(__name__) # __Name__ = __Main__ ---------> app ------------> The locality of my app /templates /static

print("Load model")
interpreter = tflite.Interpreter(model_path='ASL.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Loaded model")

IMG_SIZE = input_details[0]['shape'][1]
# Load the model

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
# Define our server and the API Paths.
@app.route('/')
def home():
    return send_file('./UX/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No Image Provided'}), 400 # BAD REQ

        # Process the image
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = img.convert('RGB')
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32) # Batch * Height * width * channels(RGB)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = classes[np.argmax(prediction)] #[0.01, 0.02....0.01] ---> 1 ()
        confidence = float(np.max(prediction))

        # Predict

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
## Load the model
## Help with Prediction

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)