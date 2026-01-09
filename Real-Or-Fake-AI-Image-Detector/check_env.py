import os
import sys

print("--- Environment Check ---")

# 1. Check Python version
print(f"Python version: {sys.version}")

# 2. Check for tensorflow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("✅ TensorFlow is installed.")
except ImportError:
    print("❌ TensorFlow is NOT installed.")
    print("   Please run: pip install -r requirements.txt")
    sys.exit(1)

# 3. Check for other dependencies
try:
    import flask
    import PIL
    import numpy
    print("✅ Flask, Pillow, and NumPy are installed.")
except ImportError as e:
    print(f"❌ Missing dependency: {e.name}")
    print("   Please run: pip install -r requirements.txt")
    sys.exit(1)


# 4. Check if model file exists
MODEL_PATH = "real_vs_ai_model.h5"
print(f"\n--- Model Check ---")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found at: {os.path.abspath(MODEL_PATH)}")
    print("   Please make sure 'real_vs_ai_model.h5' is in the main project directory.")
    sys.exit(1)
else:
    print(f"✅ Model file found at: {os.path.abspath(MODEL_PATH)}")


# 5. Try to load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
    model.summary()
except Exception as e:
    print(f"❌ Failed to load the model: {e}")
    print("   The model file might be corrupted, or there might be an issue with your TensorFlow installation.")
    sys.exit(1)

print("\n--- Conclusion ---")
print("✅ Your environment seems to be set up correctly for running the application.")
print("If the website is still not working, please check the browser's developer console (F12) for errors when you upload an image.")
