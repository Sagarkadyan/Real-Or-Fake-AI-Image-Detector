#!/usr/bin/env python3
"""
predictor.py

Contains `predict_image(image_path)` used by the Flask backend.
"""

import os
import numpy as np

# Try to import tensorflow and load the model
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as tf_image
except ImportError:
    load_model = None
    tf_image = None

# --- MODEL LOADING ---
# Load the model once at module import time.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_vs_ai_model.h5")
model = None

if load_model and tf_image and os.path.exists(MODEL_PATH):
    try:
        # Load model without compiling for faster inference
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        # The predict_image function will handle the case where the model is None

def predict_image(image_path: str) -> dict:
    """
    Run inference on the provided image path and return a dict serializable to JSON.
    Example return value: {"label": "real", "confidence": 0.9234}
    """
    # Basic validation
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"file not found: {image_path}")

    # Check if model is loaded
    if model is None:
        # This could be because tensorflow is not installed or model file is missing.
        raise RuntimeError("Model is not loaded. Please check server logs for errors (e.g., missing tensorflow or model file).")

    # --- REAL INFERENCE ---
    try:
        # Preprocess image for the model
        img = tf_image.load_img(image_path, target_size=(224, 224))
        img_array = tf_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]

        # Interpret prediction
        # In flow_from_directory, classes are sorted alphabetically:
        # "fake" = class 0, "real" = class 1
        # So prediction > 0.5 means "real" (human), prediction <= 0.5 means "fake" (AI)
        threshold = 0.5
        if prediction > threshold:
            label = "real"
            confidence = float(prediction)
        else:
            label = "fake"
            confidence = 1 - float(prediction)

        return {"label": label, "confidence": round(confidence, 4)}

    except Exception as e:
        # This will catch errors during image loading, preprocessing or prediction
        raise RuntimeError(f"Inference failed: {e}")