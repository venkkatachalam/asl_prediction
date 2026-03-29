import tensorflow as tf
import keras

# Load the Keras model
model = keras.models.load_model('ASL.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('ASL.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Done! TFLite model saved.")
print(f"Original .h5 size: {len(open('ASL.h5','rb').read()) / 1024 / 1024:.2f} MB")
print(f"TFLite size: {len(tflite_model) / 1024 / 1024:.2f} MB")
