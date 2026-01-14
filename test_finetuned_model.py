import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# --- Focal Loss (must match training one) ---
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        return K.mean(alpha * (1 - bce_exp) ** gamma * bce)
    return focal_loss_fixed

# --- Load fine-tuned model ---
model = load_model(
    "models/deepfake_detector_finetuned.keras",
    custom_objects={"focal_loss_fixed": focal_loss(2., 0.75)}
)
print("✅ Fine-tuned model loaded successfully!")

# --- Preprocess image ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- Predict ---
def predict_image(image_path):
    image = preprocess_image(image_path)
    pred = model.predict(image)[0][0]
    label = "REAL ✅" if pred < 0.5 else "FAKE ❌"
    print(f"\n📸 Image: {image_path}")
    print(f"Prediction Score: {pred:.4f}")
    print(f"Prediction: {label}")

# --- Test with sample images ---
predict_image("dataset/real/real_image.jpg")
predict_image("dataset/fake/deepfake-image-detection/Sample_fake_images/Sample_fake_images/fake/IMG-20250106-WA0013.jpg")