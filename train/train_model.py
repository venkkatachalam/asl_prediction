"""
Train a CNN to recognize ASL alphabet letters
Saves trained model as ASL.h5
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Info, Debug, Errors

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Config ---
TRAIN_DIR = os.path.join('.', 'asl_alphabet_train', 'asl_alphabet_train')
IMG_SIZE = 200
BATCH_SIZE = 32
EPOCHS = 20

# --- Data ---
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='sparse', subset='training'
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='sparse', subset='validation'
)

# --- Model ---
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'), # Hidden
    layers.Dropout(0.5), # I do not want my model to be overffit -> Neuron A is lit ---> it is 'A'. Randomly pick 50% of the total neurons in a alayer to drop.
    layers.Dense(29, activation='softmax') # OUtput -> Softmax (The prob of all classes adds upto 1)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # If my actual class is B. CNNA --> B: 0.95(Low) CNNB --> 0.1(High) -loge(prob)
model.summary()

# --- Train ---
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# --- Save ---
model.save(os.path.join('..', 'ASL.h5'))
print("Saved ASL.h5 in ASL_Final/")
