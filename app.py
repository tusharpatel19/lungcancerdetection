import os
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import tensorflow as tf

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "uploads"
MODEL_PATH = APP_ROOT / "model" / "model.h5"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}
IMG_SIZE = (224, 224)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

model = None
model_error = None


def _load_model():
    global model, model_error
    if model is not None or model_error is not None:
        return

    if not MODEL_PATH.exists():
        model_error = f"Model file not found at {MODEL_PATH}"
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        model_error = f"Failed to load model: {exc}"


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def _preprocess_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def _predict(image_path: Path):
    _load_model()
    if model is None:
        return None, model_error or "Model not available"

    x = _preprocess_image(image_path)
    preds = model.predict(x, verbose=0)
    preds = np.array(preds).squeeze()

    labels_env = os.getenv("CLASS_LABELS", "Normal,Lung Cancer,Benign,Malignant")
    labels = [label.strip() for label in labels_env.split(",") if label.strip()]

    if preds.ndim == 0:
        score = float(preds)
        label = "Lung Cancer" if score >= 0.5 else "Normal"
        confidence = score if label == "Lung Cancer" else 1.0 - score
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "cancer_probability": round(score, 4),
            "labels": ["Normal", "Lung Cancer"],
            "scores": {
                "Normal": round(1.0 - score, 4),
                "Lung Cancer": round(score, 4),
            },
        }, None

    if preds.ndim == 1 and preds.shape[0] == 1:
        score = float(preds[0])
        label = "Lung Cancer" if score >= 0.5 else "Normal"
        confidence = score if label == "Lung Cancer" else 1.0 - score
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "cancer_probability": round(score, 4),
            "labels": ["Normal", "Lung Cancer"],
            "scores": {
                "Normal": round(1.0 - score, 4),
                "Lung Cancer": round(score, 4),
            },
        }, None

    if preds.ndim == 1:
        num_classes = preds.shape[0]
        if len(labels) != num_classes:
            if num_classes == 4:
                labels = ["Normal", "Lung Cancer", "Benign", "Malignant"]
            elif num_classes == 2:
                labels = ["Normal", "Lung Cancer"]
            else:
                labels = [f"Class {i}" for i in range(num_classes)]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        return {
            "label": labels[idx],
            "confidence": round(confidence, 4),
            "scores": {labels[i]: round(float(preds[i]), 4) for i in range(num_classes)},
        }, None

    return None, "Unexpected prediction output shape"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    safe_name = "".join(ch for ch in filename if ch.isalnum() or ch in {"-", "_", "."})
    save_path = UPLOAD_DIR / safe_name
    file.save(save_path)

    result, error = _predict(save_path)
    if error:
        return jsonify({"error": error}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
