import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load your trained model
model = load_model("models/deepfake_detector_trained.h5")

# ✅ Paths to your dataset folders
folder_real = "dataset/real"
folder_fake = "dataset/fake"

# ✅ Define threshold (you can tune between 0.3 and 0.7)
threshold = 0.5

def predict_image(img_path):
    """Predict a single image and print its score"""
    if not os.path.exists(img_path):
        print(f"⚠ Image not found: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Unable to read image: {img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    score = float(model.predict(img)[0][0])
    label = "REAL ✅" if score >= threshold else "FAKE ❌"

    print(f"{os.path.basename(img_path)} → Score: {score:.4f} → {label}")

# ✅ Test on one real and one fake image
sample_real = os.path.join(folder_real, os.listdir(folder_real)[0])
sample_fake = os.path.join(folder_fake, os.listdir(folder_fake)[0])

print("\n--- Checking Model Output ---")
predict_image(sample_real)
predict_image(sample_fake)

print("\n👉 Try changing the threshold value between 0.3 and 0.7 to find the best balance.")