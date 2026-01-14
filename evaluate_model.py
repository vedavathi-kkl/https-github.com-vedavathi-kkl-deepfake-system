import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# --- Path setup ---
dataset_dir = "dataset"
model_path = "models/deepfake_detector_trained.h5"  # change this if your .h5 has a different name

# --- Load model ---
print("🔹 Loading model...")
model = load_model(model_path)
print("✅ Model loaded successfully!")

# --- Prepare data ---
print("🔹 Loading dataset...")
datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# --- Predictions ---
print("🔹 Predicting on dataset...")
preds = model.predict(test_gen)
y_pred = (preds > 0.5).astype(int).flatten()
y_true = test_gen.classes

# --- Results ---
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))