import os
import requests
from tqdm import tqdm

# Folder to save fake images
save_dir = "fake_images"
os.makedirs(save_dir, exist_ok=True)

# URL that generates random fake human faces
url = "https://thispersondoesnotexist.com/image"

# Number of images to download
num_images = 50   # change to any number you want

for i in tqdm(range(num_images)):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"fake_{i+1}.jpg")
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("Error:", e)