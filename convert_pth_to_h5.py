import torch

# Path to your .pth file
pth_model_path = r"C:\Users\Lenovo\Desktop\deepfake_system new\models"

# Load the full model
model = torch.load(pth_model_path, map_location=torch.device('cpu'))

# Set model to evaluation mode (important for inference)
model.eval()

print(model)