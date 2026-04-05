import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "model/resnet_tb_model.keras"
ct_model = load_model("tb_ct_model.h5")
IMG_SIZE = 224

# -----------------------------
# Load Model (Only Once)
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_tb(img_path):

    # Load Image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # -----------------------------
    # Stage Classification (Rule-Based)
    # -----------------------------
    if prediction < 0.5:
        label = "Normal"
        stage = "Normal"
    elif 0.5 <= prediction < 0.7:
        label = "Tuberculosis"
        stage = "Mild Tuberculosis"
    elif 0.7 <= prediction < 0.85:
        label = "Tuberculosis"
        stage = "Moderate Tuberculosis"
    else:
        label = "Tuberculosis"
        stage = "Severe Tuberculosis"

    return label, float(prediction), stage



# -----------------------------
# CT SCAN PREDICTION (128x128)
# -----------------------------
def predict_ct(img_path):

    IMG_SIZE = 128

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    prediction = ct_model.predict(img)

    if np.argmax(prediction) == 0:
        return "Normal"
    else:
        return "Tuberculosis"