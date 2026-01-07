#!/usr/bin/env python3
"""
predictor.py

Contains `predict_image(image_path)` used by the Flask backend.

By default this is a small stub that returns a random label and confidence.
Replace the stubbed logic with your model loading and inference code.

How to integrate a real model:
- If using PyTorch: load model in a module-level variable (so it loads once).
- Preprocess the image with Pillow, convert to tensor, run model, softmax, return label+confidence.
- If using TensorFlow: same idea (load once, run inference).

This file also checks optional environment variables:
- MODEL_TYPE: "pytorch" or "tensorflow" (informational, you may use it in your implementation)
- MODEL_PATH: path to the model file if you want automatic loading here.
"""

import os
import json
import random
from PIL import Image
import numpy as np

# Optional: try to import torch only if you will use it.
# import torch
# from torchvision import transforms

# Example placeholder variables for where you'd load a model:
MODEL_TYPE = os.environ.get("MODEL_TYPE", "").lower()
MODEL_PATH = os.environ.get("MODEL_PATH", "")  # e.g., "model/model.pt"

# If you plan to load a heavy model (PyTorch/TensorFlow), do it once at import time.
# Example (commented):
# model = None
# if MODEL_TYPE == "pytorch" and MODEL_PATH:
#     import torch
#     model = torch.load(MODEL_PATH, map_location="cpu")
#     model.eval()
#
# def run_pytorch_inference(image_path):
#     from PIL import Image
#     import torchvision.transforms as T
#     img = Image.open(image_path).convert("RGB")
#     preprocess = T.Compose([
#         T.Resize(256),
#         T.CenterCrop(224),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])
#     input_tensor = preprocess(img).unsqueeze(0)  # batch dimension
#     with torch.no_grad():
#         out = model(input_tensor)
#         probs = torch.nn.functional.softmax(out[0], dim=0).cpu().numpy()
#     # adapt to your label ordering
#     label_idx = int(np.argmax(probs))
#     confidence = float(np.max(probs))
#     label = "fake" if label_idx == 1 else "real"
#     return {"label": label, "confidence": round(confidence, 4)}

def predict_image(image_path: str) -> dict:
    """
    Run inference on the provided image path and return a dict serializable to JSON.
    Example return value: {"label": "real", "confidence": 0.9234}

    Replace this function's body with your real inference code.
    """
    # Basic validation
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"file not found: {image_path}")

    # Try to open image to validate it's an image
    try:
        img = Image.open(image_path).convert("RGB")
        # optionally you can do some checks: size, mode, etc.
    except Exception as e:
        raise RuntimeError(f"failed to open image: {e}")

    # -----------------------------
    # STUB INFERENCE (for demo)
    # -----------------------------
    # This returns a randomized result so the frontend and API can be tested.
    labels = ["real", "fake"]
    label = random.choice(labels)
    confidence = round(random.uniform(0.60, 0.99), 4)
    return {"label": label, "confidence": confidence}
