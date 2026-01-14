import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ---------------------------
# PATHS
# ---------------------------
model_path = "models/deepfake_detector_trained.h5"
dataset_dir = "dataset"

# ---------------------------
# LOAD MODEL
# ---------------------------
model = load_model(model_path)
print(f"✅ Loaded model from: {model_path}")

# ---------------------------
# DEFINE FOCAL LOSS
# ---------------------------
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        return K.mean(alpha * (1 - bce_exp) ** gamma * bce)
    return focal_loss_fixed

# ---------------------------
# COMPILE MODEL WITH NEW LOSS
# ---------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=focal_loss(gamma=2., alpha=0.75),
    metrics=['accuracy']
)

# ---------------------------
# DATA PREPARATION
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# ---------------------------
# TRAIN (FINE-TUNE)
# ---------------------------
print("🔁 Fine-tuning with Focal Loss to reduce fake scores...")
history = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen
)

# ---------------------------
# SAVE IMPROVED MODEL
# ---------------------------
model.save("models/deepfake_detector_finetuned.keras", save_format="keras")
print("✅ Model saved as models/deepfake_detector_finetuned.h5")