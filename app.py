from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
from model.predictor import predict_image

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}

def allowed_filename(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXT

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    # serve static assets from public/
    return send_from_directory(app.static_folder, filename)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_filename(file.filename):
        return jsonify({"error": "file type not allowed"}), 400

    filename = secure_filename(file.filename)
    # use a temporary file for safe handling
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="upload-", suffix=os.path.splitext(filename)[1], dir=UPLOAD_DIR)
    os.close(tmp_fd)
    try:
        file.save(tmp_path)
        # call predictor (should return a JSON-serializable dict)
        result = predict_image(tmp_path)
        return jsonify(result)
    except Exception as e:
        # don't leak internal trace to the client, but return message
        return jsonify({"error": "inference failed", "details": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    # Development server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
