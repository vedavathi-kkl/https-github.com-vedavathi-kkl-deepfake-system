import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Paths
model_path = "models/deepfake_detector_trained.h5"
dataset_dir = "dataset"   # only one folder having 'fake' and 'real' inside

# Load model
model = load_model(model_path)
print(f"✅ Loaded model from: {model_path}")

from tensorflow.keras import backend as K
import tensorflow as tf

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        return K.mean(alpha * (1 - bce_exp) ** gamma * bce)
    return focal_loss_fixed

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(gamma=2., alpha=0.75),
    metrics=['accuracy']
)

# Data generator
datagen = ImageDataGenerator(rescale=1.0/255)

# Load dataset
data_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print(f"✅ Class Indices: {data_gen.class_indices}")

# Predictions
print("🔍 Evaluating on dataset...")
preds = model.predict(data_gen, verbose=1)
labels = data_gen.classes
pred_classes = (preds > 0.5).astype(int)

# Accuracy
acc = accuracy_score(labels, pred_classes)
print(f"✅ Overall Accuracy: {acc*100:.2f}%")

# Confusion Matrix and Report
print("\n📊 Confusion Matrix:")
print(confusion_matrix(labels, pred_classes))

print("\n📈 Classification Report:")
print(classification_report(labels, pred_classes, target_names=['Fake', 'Real']))

# Optional: Save predictions
results = pd.DataFrame({
    "Filename": data_gen.filenames,
    "True Label": labels,
    "Predicted": pred_classes.flatten(),
    "Score": preds.flatten()
})
results.to_csv("dataset_results.csv", index=False)
print("✅ Results saved to dataset_results.csv")

print("\n🎯 Completed successfully!")