import os
import shutil
import random

base_dir = 'dataset'
real_dir = os.path.join(base_dir, 'real')
fake_dir = os.path.join(base_dir, 'fake')

train_real = os.path.join(base_dir, 'train/real')
train_fake = os.path.join(base_dir, 'train/fake')
val_real   = os.path.join(base_dir, 'val/real')
val_fake   = os.path.join(base_dir, 'val/fake')

os.makedirs(train_real, exist_ok=True)
os.makedirs(train_fake, exist_ok=True)
os.makedirs(val_real, exist_ok=True)
os.makedirs(val_fake, exist_ok=True)

def get_all_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, f))
    return images

def split_data(source, train_target, val_target, ratio=0.8):
    files = get_all_images(source)
    random.shuffle(files)
    split = int(len(files)*ratio)
    train_files = files[:split]
    val_files = files[split:]

    for f in train_files:
        shutil.copy(f, train_target)
    for f in val_files:
        shutil.copy(f, val_target)
    print(f"{source} → Train: {len(train_files)}, Val: {len(val_files)}")

split_data(real_dir, train_real, val_real)
split_data(fake_dir, train_fake, val_fake)
print("✅ Dataset split complete.")