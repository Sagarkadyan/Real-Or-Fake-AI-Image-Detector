# Real-Or-Fake AI Image Detector — Python Backend + Static Frontend

This repo now contains a Python (Flask) backend that serves a simple static frontend and a prediction endpoint.

What you get
- Flask app (app.py) serves `public/` and exposes:
  - GET /api/health
  - POST /api/predict (multipart form-data, field name `image`)
- A Python predictor stub at `model/predictor.py` that you should replace with your real model code.
- Static frontend in `public/` (index.html, style.css, app.js) — same UI as before.
- `requirements.txt` for installing dependencies.

Quick start (development)
1. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) If you use PyTorch or TensorFlow, install those as needed:
   ```
   pip install torch torchvision       # or follow torch install instructions for your platform
   pip install tensorflow             # if using TF
   ```

4. Run the server:
   ```
   python app.py
   ```
   The app will be available at http://localhost:5000

API
- POST /api/predict
  - Form field: `image` (file)
  - Response: JSON, example: `{"label":"real", "confidence":0.9234}`

Integrating your real model
- Edit `model/predictor.py`:
  - Load your model once at module import (so it's not reloaded per request).
  - Add a function to preprocess images (Pillow -> tensor / array).
  - Run the model and return a JSON-serializable dict: at minimum `label` and `confidence`.
- See comments in `model/predictor.py` for a PyTorch example snippet.

Deployment notes
- For production, run with a WSGI server (gunicorn/uvicorn) and consider:
  - Limiting upload sizes (Flask config: MAX_CONTENT_LENGTH).
  - Adding authentication / rate-limiting.
  - Running heavy model inference on GPU-backed machines or a separate microservice.
  - Using cloud storage for uploads, or streaming input into the model without saving to disk.

If you want, I can:
- Replace the predictor stub with a concrete PyTorch inference implementation if you provide the model type (PyTorch/TF) and the model file (or its expected path).
- Add a dockerfile and docker-compose to build an image that includes CPU PyTorch or TF.
