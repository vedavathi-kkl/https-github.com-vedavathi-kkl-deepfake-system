import os

real_path = "dataset/real"
fake_path = "dataset/fake"

real_count = len(os.listdir(real_path))
fake_count = len(os.listdir(fake_path))

print(f"🟢 Real images: {real_count}")
print(f"🔴 Fake images: {fake_count}")
print(f"📸 Total images: {real_count + fake_count}")