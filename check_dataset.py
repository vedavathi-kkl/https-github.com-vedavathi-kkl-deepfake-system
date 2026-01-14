import os

real_dir = "dataset/real"
fake_dir = "dataset/fake"

print("Real images:", len(os.listdir(real_dir)))
print("Fake images:", len(os.listdir(fake_dir)))