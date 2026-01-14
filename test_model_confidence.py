import tensorflow as tf
import numpy as np
import cv2
import os

# =========================================
# LOAD TRAINED MODEL
# =========================================
model_path = "models/deepfake_detector_v2.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}")
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded successfully from {model_path}")

IMG_SIZE = 224

# =========================================
# FUNCTION TO PREDICT IMAGE
# =========================================
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠ Could not load image: {image_path}")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prob = float(model.predict(img, verbose=0)[0][0])
    label = "Real" if prob >= 0.5 else "Fake"

    # Confidence Calculation
    confidence = prob if label == "Real" else (1 - prob)
    confidence_pct = round(confidence * 100, 2)
    level = "High" if confidence_pct >= 85 else "Medium" if confidence_pct >= 65 else "Low"

    print(f"🖼 {os.path.basename(image_path)} → Prediction: {label} | Confidence: {confidence_pct}% ({level})")
    return label, confidence_pct

# =========================================
# TEST EXAMPLES
# =========================================
# Place 3 images here to test:
# 1. Real untouched
# 2. Cropped/edited real
# 3. AI-generated fake

test_images = [
    "test_images/real_original.jpg",
    "test_images/real_cropped.jpg",
    "test_images/fake_ai.jpg"
]

for path in test_images:
    predict_image(path)