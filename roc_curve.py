import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE = 224
MODEL_PATH = "model/resnet_tb_model.keras"
DATASET_DIR = "dataset"

# ---------------------------
# Load Model
# ---------------------------
model = load_model(MODEL_PATH)

# ---------------------------
# Load Validation Data
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ---------------------------
# Get True Labels & Predictions
# ---------------------------
y_true = validation_generator.classes
y_pred_prob = model.predict(validation_generator).ravel()

# ---------------------------
# ROC Calculation
# ---------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

# ---------------------------
# Plot ROC Curve
# ---------------------------
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - TB Detection Model")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()

print(f"AUC Score: {roc_auc:.4f}")