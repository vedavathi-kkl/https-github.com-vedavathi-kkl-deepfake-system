import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 8
EPOCHS_FINE = 25   # ⬆ longer fine-tuning

train_dir = "dataset/train"
val_dir   = "dataset/val"

# ---- LIGHTER AUGMENTATION (better realism) ----
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=6,
    width_shift_range=0.06,
    height_shift_range=0.06,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode="binary"
)
val_ds = val_gen.flow_from_directory(
    val_dir, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode="binary"
)

# ---- BUILD MODEL ----
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = EfficientNetB0(include_top=False, weights="imagenet",
                             input_tensor=inputs, pooling="avg")
base_model.trainable = False

x = layers.Dropout(0.3)(base_model.output)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])

print("\n🔹 Stage 1: Training top layers ...")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# ---- FINE-TUNE DEEPER LAYERS ----
print("\n🔹 Stage 2: Fine-tuning more layers ...")
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics=["accuracy"])

history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

# ---- SAVE MODEL ----
os.makedirs("models", exist_ok=True)
model.save("models/deepfake_detector_v3.h5")
print("\n✅ Model saved as models/deepfake_detector_v3.h5")

# ---- PLOT TRAINING ----
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.legend(); plt.title("Training vs Validation Accuracy"); plt.show()