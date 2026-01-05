import random
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "real_vs_ai_model.h5"
TEST_REAL_DIR = "dataset/real"
TEST_FAKE_DIR = "dataset/fake"
TEST_SAMPLES = 20  # Number of samples to test from each class

# -----------------------
# LOAD MODEL
# -----------------------
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    print("Please train the model first by running: python train.py")
    exit(1)

print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# -----------------------
# PREPARE TEST DATA
# -----------------------


def get_test_images(directory, num_samples):
    """Get random test images from directory"""
    images = [f for f in os.listdir(directory)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) > num_samples:
        return random.sample(images, num_samples)
    return images


real_images = get_test_images(TEST_REAL_DIR, TEST_SAMPLES)
fake_images = get_test_images(TEST_FAKE_DIR, TEST_SAMPLES)

print(
    f"📊 Testing with {len(real_images)} real images and {len(fake_images)} fake images\n")

# -----------------------
# TEST FUNCTION
# -----------------------


def predict_image(model, image_path):
    """Predict if image is AI generated or human prepared"""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    # Note: "fake" = class 0, "real" = class 1 (alphabetical order)
    # prediction > 0.5 = class 1 = "real" = Human Prepared
    # prediction <= 0.5 = class 0 = "fake" = AI Generated
    # True = AI Generated (class 0), False = Human Prepared (class 1)
    return prediction <= 0.5  # Return True if AI Generated, False if Human Prepared


# -----------------------
# RUN TESTS
# -----------------------
y_true = []
y_pred = []

print("🧪 Testing real images (should be 'Prepared by human')...")
for img_name in real_images:
    img_path = os.path.join(TEST_REAL_DIR, img_name)
    pred = predict_image(model, img_path)
    y_true.append(0)  # 0 = Human Prepared
    y_pred.append(1 if pred else 0)

print("🧪 Testing fake images (should be 'AI Generated Image')...")
for img_name in fake_images:
    img_path = os.path.join(TEST_FAKE_DIR, img_name)
    pred = predict_image(model, img_path)
    y_true.append(1)  # 1 = AI Generated
    y_pred.append(1 if pred else 0)

# -----------------------
# RESULTS
# -----------------------
accuracy = accuracy_score(y_true, y_pred)
print("\n" + "=" * 50)
print("📊 TEST RESULTS")
print("=" * 50)
print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred,
                            target_names=['Prepared by human', 'AI Generated Image']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print("                    Predicted")
print("                  Human    AI")
print(f"Actual Human    {cm[0][0]:4d}  {cm[0][1]:4d}")
print(f"Actual AI       {cm[1][0]:4d}  {cm[1][1]:4d}")
print("=" * 50)
