# filepath: c:\VS Language Codes\real vs ai model\train.py
import os
import kagglehub  # Add this import
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# -----------------------
# CONFIG
# -----------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6

# Download dataset if not present
if not os.path.exists("dataset"):
    print("Downloading dataset...")
    path = kagglehub.dataset_download(
        "cashbowman/ai-generated-images-vs-real-images")
    # Assuming the dataset has 'real' and 'ai' subfolders; adjust if needed
    os.rename(path, "dataset")  # Rename the downloaded folder to 'dataset'

DATASET_PATH = "dataset"

# ...existing code...
