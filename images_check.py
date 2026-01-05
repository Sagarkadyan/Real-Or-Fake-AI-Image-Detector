import os

print("REAL images:", len(os.listdir("dataset/real")))
print("FAKE images:", len(os.listdir("dataset/fake")))
