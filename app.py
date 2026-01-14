# =============================================================
# ✅ DeepShield AI - Deepfake Detection Flask App
# =============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import binary_crossentropy
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2

# -------------------------------------------------------------
# ✅ Custom Focal Loss (Registered for Model Loading)
# -------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="Custom")
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = binary_crossentropy(y_true, y_pred)
    bce_exp = tf.exp(-bce)
    loss = alpha * (1 - bce_exp) ** gamma * bce
    return tf.reduce_mean(loss)

# -------------------------------------------------------------
# ✅ Flask Setup
# -------------------------------------------------------------
app = Flask(__name__)

# Folder for uploads
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------------------------------------
# ✅ Load Trained Model
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_detector_finetuned.keras")

try:
    model = load_model(MODEL_PATH, custom_objects={"focal_loss_fixed": focal_loss_fixed})
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# -------------------------------------------------------------
# ✅ Image Preprocessing Function
# -------------------------------------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image format or path")
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------------------------------------
# 🌐 FRONTEND ROUTES
# -------------------------------------------------------------

@app.route("/")
def landing():
    """Landing (Welcome) page"""
    return render_template("landing.html", title="DeepShield AI")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page (placeholder)"""
    if request.method == "POST":
        return redirect(url_for("upload"))
    return render_template("login.html", title="Login - DeepShield AI")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload image for deepfake detection"""
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            return redirect(url_for("predict", filename=filename))

    return render_template("upload.html", title="Upload - DeepShield AI")

@app.route("/predict/<filename>")
def predict(filename):
    """Run model prediction and show result"""
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if model is None:
        return render_template(
            "result.html",
            title="Result - DeepShield AI",
            result_text="Model not loaded",
            confidence=0,
            image_path=None
        )

    try:
        image = preprocess_image(filepath)
        prediction = model.predict(image)[0][0]
        print("Raw prediction:", prediction)
    except Exception as e:
        return render_template(
            "result.html",
            title="Result - DeepShield AI",
            result_text=f"Error processing: {e}",
            confidence=0,
            image_path=None
        )

    # ✅ Binary classification logic
    if prediction < 0.5:
        result_text = "⚠ Fake Image Detected"
        confidence = round(float((1 - prediction) * 100), 2)
    else:
        result_text = "✅ Real Image Detected"
        confidence = round(float(prediction * 100), 2)

    return render_template(
        "result.html",
        title="Result - DeepShield AI",
        result_text=result_text,
        confidence=confidence,
        image_path=f"uploads/{filename}"
    )

# -------------------------------------------------------------
# ✅ Run Flask App
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
