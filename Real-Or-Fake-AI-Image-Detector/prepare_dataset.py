import os
import random
import shutil
import sys
import kagglehub

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

IMAGES_PER_CLASS = 300  # Increased for better model performance
TARGET_DIR = "dataset"

# Download dataset
source_path = kagglehub.dataset_download(
    "cashbowman/ai-generated-images-vs-real-images"
)

print("Dataset path:", source_path)

REAL_SOURCE = "RealArt"
FAKE_SOURCE = "AiArtData"

os.makedirs("dataset/real", exist_ok=True)
os.makedirs("dataset/fake", exist_ok=True)


def collect_images(root_folder):
    image_files = []
    for root, _, files in os.walk(root_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, f))
    return image_files


def copy_images(src_folder, dst_folder):
    src_path = os.path.join(source_path, src_folder)
    dst_path = os.path.join(TARGET_DIR, dst_folder)

    images = collect_images(src_path)

    if len(images) == 0:
        raise Exception(f"❌ No images found in {src_path}")

    selected = random.sample(images, min(IMAGES_PER_CLASS, len(images)))

    for img in selected:
        shutil.copy(img, dst_path)

    print(f"✅ Copied {len(selected)} images to {dst_path}")


copy_images(REAL_SOURCE, "real")
copy_images(FAKE_SOURCE, "fake")

print("\n🎉 Dataset prepared successfully!")
