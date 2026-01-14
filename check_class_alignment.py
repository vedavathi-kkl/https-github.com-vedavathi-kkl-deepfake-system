from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 🔹 Load your model
model = load_model("models/deepfake_detector_trained.h5")

# 🔹 Check if model has class names stored
print("Model class indices:", getattr(model, "class_names", "⚠ No class_names attribute found"))

# 🔹 Load dataset to see its class mapping
data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(
    "dataset",        # ✅ change this if your dataset folder name is different
    target_size=(224, 224)
)

print("Dataset class indices:", data_flow.class_indices)