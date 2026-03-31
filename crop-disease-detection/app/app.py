"""
app.py
======
Flask web application for the Crop Disease Detection System.

Routes:
    GET  /           → Upload page (index.html)
    POST /predict    → Accepts image upload, returns JSON prediction result
    GET  /health     → Health check endpoint

Usage:
    python app/app.py
    Open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import json
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Add src/ directory to path so we can import predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_image, get_advice

# ──────────────────────────────────────────────
# Flask app setup
# ──────────────────────────────────────────────
app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

app.config["SECRET_KEY"]       = "crop-disease-detection-2025"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload size
app.config["UPLOAD_FOLDER"]    = os.path.join(os.path.dirname(__file__), "uploads")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Create uploads folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def model_path() -> str:
    """Return absolute path to the model pickle file."""
    return os.path.join(
        os.path.dirname(__file__), "..", "models", "decision_tree_model.pkl"
    )


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    """
    Render the main upload page.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a leaf image upload and return a JSON prediction result.

    Request:
        multipart/form-data with field 'image' containing the image file.

    Response JSON:
        {
            "success":          true,
            "predicted_class":  "Leaf_Spot",
            "confidence":       94.5,
            "all_probabilities": {...},
            "decision_rules":   [...],
            "feature_values":   {...},
            "advice": {
                "description": "...",
                "action":      "...",
                "severity":    "Moderate"
            }
        }
    """
    # ── Validate request ──
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # ── Save uploaded file temporarily ──
    ext       = file.filename.rsplit(".", 1)[1].lower()
    unique_fn = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_fn)
    file.save(save_path)

    try:
        # ── Run prediction pipeline ──
        result = predict_image(save_path, model_path())
        advice = get_advice(result["predicted_class"])

        response = {
            "success":           True,
            "predicted_class":   result["predicted_class"],
            "confidence":        result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "decision_rules":    result["decision_rules"],
            "feature_values":    result["feature_values"],
            "advice":            advice,
        }
        return jsonify(response), 200

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # ── Always clean up the temporary uploaded file ──
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/health")
def health():
    """Health check endpoint."""
    model_exists = os.path.exists(model_path())
    return jsonify({
        "status":       "ok" if model_exists else "model_missing",
        "model_loaded": model_exists,
        "model_path":   model_path()
    })


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Crop Disease Detection — Flask Web App")
    print("=" * 55)

    if not os.path.exists(model_path()):
        print(f"\n[WARNING] Model not found at: {model_path()}")
        print("          Run src/train.py first to generate the model.")
    else:
        print(f"\n[OK] Model found: {model_path()}")

    print("\n[INFO] Starting server at http://127.0.0.1:5000")
    print("       Press CTRL+C to stop.\n")

    app.run(debug=True, host="127.0.0.1", port=5000)
