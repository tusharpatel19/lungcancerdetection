import os
import csv
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import wraps
from io import BytesIO
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

from agent_service import run_agentic_analysis

try:
    from flask_cors import CORS
except Exception:  # pragma: no cover
    def CORS(*_args, **_kwargs):
        return None

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "uploads"
DEFAULT_MODEL_CANDIDATES = [
    APP_ROOT / "model" / "model.h5",
    APP_ROOT / "model" / "model.keras",
    Path("/var/data/model.h5"),
    Path("/var/data/model.keras"),
]
MODEL_PATH_ENV = os.getenv("MODEL_PATH", "").strip()
MODEL_URL = os.getenv("MODEL_URL", "").strip()
DB_PATH = APP_ROOT / "predictions.db"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "5242880"))
IMG_SIZE = (224, 224)
CLASS_LABELS_FILE = APP_ROOT / "model" / "class_labels.txt"
CONFUSION_MATRIX_FILE = APP_ROOT / "model" / "confusion_matrix.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(APP_ROOT / "public"),
    static_folder=str(APP_ROOT / "public" / "static"),
    static_url_path="/static",
)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = os.getenv("FLASK_SECRET_KEY", "lungguard-dev-secret")

DEFAULT_CORS_ORIGINS = "*"
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if origin.strip()]
REQUIRE_API_LOGIN = os.getenv("REQUIRE_API_LOGIN", "false").strip().lower() in {"1", "true", "yes", "on"}

CORS(
    app,
    resources={
        r"/predict": {"origins": CORS_ORIGINS},
        r"/save_prediction": {"origins": CORS_ORIGINS},
        r"/history": {"origins": CORS_ORIGINS},
        r"/database": {"origins": CORS_ORIGINS},
        r"/download_report": {"origins": CORS_ORIGINS},
        r"/healthz": {"origins": CORS_ORIGINS},
        r"/api/analyze": {"origins": CORS_ORIGINS},
        r"/api/analysis/*": {"origins": CORS_ORIGINS},
    },
)


@app.after_request
def add_no_cache_headers(response):
    if request.path.startswith("/static/") or request.path in {"/", "/index.html"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


model = None
model_error = None
resolved_model_path = None


def _restore_model_from_parts() -> Path | None:
    model_dir = APP_ROOT / "model"
    parts = sorted(model_dir.glob("model.h5.part*"))
    if not parts:
        return None

    target = model_dir / "model.h5"
    if target.exists():
        return target

    try:
        with target.open("wb") as out:
            for part in parts:
                with part.open("rb") as src:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
        return target
    except Exception:
        if target.exists():
            target.unlink()
        return None


def _resolve_model_path() -> Path | None:
    candidates: list[Path] = []
    if MODEL_PATH_ENV:
        candidates.append(Path(MODEL_PATH_ENV))
    candidates.extend(DEFAULT_MODEL_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    restored = _restore_model_from_parts()
    if restored is not None:
        return restored
    return None


def _ensure_model_from_url() -> Path | None:
    if not MODEL_URL:
        return None

    model_dir = APP_ROOT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "downloaded_model.h5"
    if target.exists():
        return target

    try:
        urlretrieve(MODEL_URL, target)  # noqa: S310
        return target
    except Exception:
        return None


def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                prediction_result TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                image_path TEXT,
                explanation TEXT,
                research TEXT,
                sources TEXT,
                report TEXT,
                model_version TEXT,
                status TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT,
                prediction TEXT,
                confidence REAL,
                model_version TEXT,
                explanation TEXT,
                research TEXT,
                sources TEXT,
                report TEXT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'initialized'
            )
            """
        )
        conn.commit()


def _login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def _require_api_login():
    if not REQUIRE_API_LOGIN:
        return None
    if not session.get("user_id"):
        return jsonify({"error": "Unauthorized. Please login first."}), 401
    return None


def _prediction_rows(limit: int = 100) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, patient_name, prediction_result, confidence, risk_level, image_path, explanation, research, sources, report, model_version, status, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]

def _analysis_rows(limit: int = 20) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, user_id, image_path, prediction, confidence, model_version, explanation, research, sources, report, created_at, status
            FROM analysis
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = []
    for row in rows:
        item = dict(row)
        for key in {"explanation", "research", "sources"}:
            value = item.get(key)
            if value:
                try:
                    item[key] = json.loads(value)
                except (TypeError, ValueError):
                    item[key] = value
        result.append(item)
    return result


def _analysis_by_id(analysis_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, user_id, image_path, prediction, confidence, model_version, explanation, research, sources, report, created_at, status
            FROM analysis
            WHERE id = ?
            """,
            (analysis_id,),
        ).fetchone()
    if row is None:
        return None

    item = dict(row)
    for key in {"explanation", "research", "sources"}:
        value = item.get(key)
        if value:
            try:
                item[key] = json.loads(value)
            except (TypeError, ValueError):
                item[key] = value
    return item


def _save_analysis_record(state) -> dict | None:
    if not state or not getattr(state, "prediction", None):
        return None

    now = datetime.now(timezone.utc).isoformat()
    explanation = json.dumps(getattr(state, "explanation", {}) or {}, ensure_ascii=False)
    research = json.dumps(getattr(state, "research", {}) or {}, ensure_ascii=False)
    sources = json.dumps(getattr(state, "sources", []) or [], ensure_ascii=False)
    report = getattr(state, "report", "") or ""

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO analysis (user_id, image_path, prediction, confidence, model_version, explanation, research, sources, report, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                None,
                getattr(state, "imagePath", None),
                getattr(state, "prediction", None),
                getattr(state, "confidence", None),
                getattr(state, "modelVersion", "v1"),
                explanation,
                research,
                sources,
                report,
                now,
                getattr(state, "status", "initialized"),
            ),
        )
        analysis_id = cursor.lastrowid
        conn.commit()

    return _analysis_by_id(analysis_id)

def _load_model():
    global model, model_error, resolved_model_path
    if model is not None or model_error is not None:
        return

    model_path = _resolve_model_path()
    if model_path is None:
        model_path = _ensure_model_from_url()
    if model_path is None:
        searched = [str(p) for p in ([Path(MODEL_PATH_ENV)] if MODEL_PATH_ENV else []) + DEFAULT_MODEL_CANDIDATES]
        model_error = (
            "Model file not found. Set MODEL_PATH or MODEL_URL. "
            f"Searched: {', '.join(searched)}"
        )
        return

    try:
        # Lazy import keeps server startup fast and prevents hard crash if TensorFlow install fails.
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        model = tf.keras.models.load_model(model_path)
        resolved_model_path = model_path
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


def _looks_like_ct_scan(image_path: Path) -> tuple[bool, str]:
    try:
        image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    except Exception:
        return False, "Uploaded file is not a readable image."

    arr = np.asarray(image, dtype=np.float32)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    mean_saturation = float(hsv[:, :, 1].mean())
    mean_channel_delta = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    dark_pixel_ratio = float((arr.mean(axis=2) < 35).mean())

    if mean_saturation > 25 or mean_channel_delta > 12:
        return False, "Please upload a grayscale lung CT scan image, not a normal photo."
    if dark_pixel_ratio < 0.05:
        return False, "This image does not look like a lung CT scan. Please upload a valid CT slice."
    return True, ""


def _labels_from_env() -> list[str]:
    labels_env = os.getenv("CLASS_LABELS", "").strip()
    if not labels_env:
        return []
    return [label.strip() for label in labels_env.split(",") if label.strip()]


def _labels_from_file() -> list[str]:
    if not CLASS_LABELS_FILE.exists():
        return []
    return [line.strip() for line in CLASS_LABELS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _labels_from_confusion_matrix() -> list[str]:
    if not CONFUSION_MATRIX_FILE.exists():
        return []
    try:
        with CONFUSION_MATRIX_FILE.open(newline="", encoding="utf-8") as file:
            header = next(csv.reader(file), [])
    except Exception:
        return []
    return [label.strip() for label in header[1:] if label.strip()]


def _labels_from_training_dirs() -> list[str]:
    train_dir = APP_ROOT / "data" / "Data" / "train"
    if not train_dir.exists():
        return []
    return sorted(path.name for path in train_dir.iterdir() if path.is_dir())


def _class_labels(num_classes: int) -> list[str]:
    for labels in (_labels_from_env(), _labels_from_file(), _labels_from_confusion_matrix(), _labels_from_training_dirs()):
        if len(labels) == num_classes:
            return labels

    if num_classes == 2:
        return ["Normal", "Lung Cancer"]
    return [f"Class {i}" for i in range(num_classes)]


def _display_label(label: str) -> str:
    text = label.replace("_", " ").replace(".", " ").strip()
    lower = text.lower()
    if lower == "normal":
        return "Normal"
    if "adenocarcinoma" in lower:
        return "Adenocarcinoma"
    if "large" in lower and "cell" in lower:
        return "Large Cell Carcinoma"
    if "squamous" in lower and "cell" in lower:
        return "Squamous Cell Carcinoma"
    if "malignant" in lower:
        return "Malignant"
    if "benign" in lower:
        return "Benign"
    if "lung" in lower and "cancer" in lower:
        return "Lung Cancer"
    return " ".join(part.capitalize() for part in text.split()) or label


def _is_cancer_label(label: str) -> bool:
    lower = label.lower()
    if "normal" in lower or "benign" in lower:
        return False
    cancer_terms = ("cancer", "carcinoma", "adenocarcinoma", "squamous", "malignant")
    return any(term in lower for term in cancer_terms)


def _predict(image_path: Path):
    _load_model()
    if model is None:
        return None, model_error or "Model not available"

    x = _preprocess_image(image_path)
    preds = model.predict(x, verbose=0)
    preds = np.array(preds).squeeze()

    if preds.ndim == 0:
        score = float(preds)
        label = "Lung Cancer" if score >= 0.5 else "Normal"
        confidence = score if label == "Lung Cancer" else 1.0 - score
        return {
            "label": label,
            "display_label": _display_label(label),
            "is_cancer": _is_cancer_label(label),
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
            "display_label": _display_label(label),
            "is_cancer": _is_cancer_label(label),
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
        labels = _class_labels(num_classes)
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        cancer_probability = float(
            sum(float(preds[i]) for i, label in enumerate(labels) if _is_cancer_label(label))
        )
        return {
            "label": labels[idx],
            "display_label": _display_label(labels[idx]),
            "is_cancer": _is_cancer_label(labels[idx]),
            "confidence": round(confidence, 4),
            "cancer_probability": round(cancer_probability, 4),
            "scores": {_display_label(labels[i]): round(float(preds[i]), 4) for i in range(num_classes)},
        }, None

    return None, "Unexpected prediction output shape"


def _risk_level(label: str, confidence: float, cancer_probability: float | None) -> str:
    lower = label.lower()
    if "normal" in lower or "benign" in lower:
        return "Low"

    signal = cancer_probability if cancer_probability is not None else confidence
    if signal >= 0.9:
        return "Critical"
    if signal >= 0.75:
        return "High"
    if signal >= 0.5:
        return "Moderate"
    return "Low"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(lines: list[str]) -> bytes:
    content_lines = ["BT", "/F1 12 Tf", "50 790 Td"]
    first = True
    for line in lines:
        if first:
            content_lines.append(f"({_pdf_escape(line)}) Tj")
            first = False
        else:
            content_lines.append("T*")
            content_lines.append(f"({_pdf_escape(line)}) Tj")
    content_lines.append("ET")
    content = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(
        f"5 0 obj << /Length {len(content)} >> stream\n".encode("latin-1")
        + content
        + b"\nendstream endobj\n"
    )

    output = BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(output.tell())
        output.write(obj)

    xref_start = output.tell()
    output.write(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    output.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.write(f"{off:010d} 00000 n \n".encode("latin-1"))
    output.write(
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode(
            "latin-1"
        )
    )
    return output.getvalue()


@app.route("/")
def index():
    return render_template("index.html", user_name=session.get("user_name", "User"))


@app.route("/freelance")
def freelance():
    return render_template("freelance.html", user_name=session.get("user_name", "User"))


@app.route("/signup", methods=["GET", "POST"])
@app.route("/signup.html", methods=["GET", "POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = ""
    if request.method == "POST":
        full_name = str(request.form.get("full_name", "")).strip()
        email = str(request.form.get("email", "")).strip().lower()
        password = str(request.form.get("password", ""))
        confirm_password = str(request.form.get("confirm_password", ""))

        if not full_name or not email or not password:
            error = "All fields are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            with sqlite3.connect(DB_PATH) as conn:
                existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
                if existing:
                    error = "Account already exists with this email."
                else:
                    conn.execute(
                        """
                        INSERT INTO users (full_name, email, password_hash, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (full_name, email, generate_password_hash(password), datetime.now(timezone.utc).isoformat()),
                    )
                    conn.commit()
                    return redirect(url_for("login"))

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
@app.route("/login.html", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = ""
    if request.method == "POST":
        email = str(request.form.get("email", "")).strip().lower()
        password = str(request.form.get("password", ""))

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            user = conn.execute(
                "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()

        if not user or not check_password_hash(user["password_hash"], password):
            error = "Invalid email or password."
        else:
            session["user_id"] = user["id"]
            session["user_name"] = user["full_name"]
            session["user_email"] = user["email"]
            return redirect(url_for("index"))

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/healthz", methods=["GET"])
def healthz():
    model_path = _resolve_model_path()
    parts_count = len(list((APP_ROOT / "model").glob("model.h5.part*")))
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "model_exists": model_path is not None,
            "model_error": model_error,
            "model_path": str(resolved_model_path or model_path or ""),
            "model_path_env": MODEL_PATH_ENV,
            "model_url_set": bool(MODEL_URL),
            "model_parts_found": parts_count,
            "api_login_required": REQUIRE_API_LOGIN,
        }
    )


@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded", "status": "invalid_image"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename", "status": "invalid_image"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type", "status": "invalid_image"}), 400

    try:
        if hasattr(file, "stream"):
            file.stream.seek(0, os.SEEK_END)
            size = file.stream.tell()
            file.stream.seek(0)
            if size > MAX_UPLOAD_BYTES:
                return jsonify({"error": f"File exceeds the {MAX_UPLOAD_BYTES} byte limit", "status": "invalid_image"}), 400
    except Exception:
        pass

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    safe_name = "".join(ch for ch in filename if ch.isalnum() or ch in {"-", "_", "."})
    save_path = UPLOAD_DIR / safe_name
    file.save(save_path)

    if save_path.stat().st_size > MAX_UPLOAD_BYTES:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": f"File exceeds the {MAX_UPLOAD_BYTES} byte limit", "status": "invalid_image"}), 400

    is_ct_scan, ct_error = _looks_like_ct_scan(save_path)
    if not is_ct_scan:
        return jsonify({"error": ct_error, "status": "invalid_image"}), 400

    state = run_agentic_analysis(str(save_path))
    _save_analysis_record(state)

    if state.validationError:
        return jsonify({
            "error": state.validationError,
            "status": "invalid_image",
            "prediction": state.prediction,
            "confidence": state.confidence,
            "modelVersion": state.modelVersion,
        }), 400

    response = {
        "status": state.status,
        "imagePath": state.imagePath,
        "prediction": state.prediction,
        "confidence": state.confidence,
        "modelVersion": state.modelVersion,
        "explanation": state.explanation,
        "research": state.research,
        "sources": state.sources,
        "report": state.report,
    }
    return jsonify(response)


@app.route("/api/analysis/history", methods=["GET"])
def analysis_history_api():
    return jsonify({"items": _analysis_rows(limit=20)})


@app.route("/api/analysis/<int:analysis_id>", methods=["GET"])
def analysis_detail_api(analysis_id: int):
    item = _analysis_by_id(analysis_id)
    if item is None:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify(item)


@app.route("/predict", methods=["POST"])
def predict():
    auth_error = _require_api_login()
    if auth_error:
        return auth_error

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

    is_ct_scan, ct_error = _looks_like_ct_scan(save_path)
    if not is_ct_scan:
        return jsonify({"error": ct_error}), 400

    result, error = _predict(save_path)
    if error:
        return jsonify({"error": error}), 500

    cancer_prob = result.get("cancer_probability")
    risk = _risk_level(result.get("label", ""), float(result.get("confidence", 0)), cancer_prob)
    result["risk_level"] = risk
    result["image_path"] = f"/uploads/{safe_name}"
    return jsonify(result)


@app.route("/save_prediction", methods=["POST"])
def save_prediction():
    auth_error = _require_api_login()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True) or {}
    patient_name = str(payload.get("patient_name", "")).strip()
    prediction_result = str(payload.get("prediction_result", "")).strip()
    confidence = payload.get("confidence")
    risk_level = str(payload.get("risk_level", "")).strip()
    image_path = str(payload.get("image_path", "")).strip()

    if not patient_name:
        return jsonify({"error": "Patient name is required"}), 400
    if not prediction_result:
        return jsonify({"error": "Prediction result is required"}), 400
    if confidence is None:
        return jsonify({"error": "Confidence is required"}), 400
    if not risk_level:
        return jsonify({"error": "Risk level is required"}), 400

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                patient_name, prediction_result, confidence, risk_level, image_path, explanation, research, sources, report, model_version, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_name,
                prediction_result,
                float(confidence),
                risk_level,
                image_path,
                "",
                "",
                "[]",
                "",
                "v1",
                "saved",
                now,
            ),
        )
        conn.commit()

    return jsonify({"message": "Prediction saved successfully"})


@app.route("/history", methods=["GET"])
def history():
    auth_error = _require_api_login()
    if auth_error:
        return auth_error

    return jsonify({"items": _prediction_rows(limit=20)})


@app.route("/database", methods=["GET"])
def database():
    auth_error = _require_api_login()
    if auth_error:
        return auth_error

    return jsonify(
        {
            "database": str(DB_PATH),
            "tables": {
                "predictions": {
                    "columns": ["id", "patient_name", "prediction_result", "confidence", "risk_level", "image_path", "created_at"],
                    "rows": _prediction_rows(limit=100),
                }
            },
        }
    )


@app.route("/download_report", methods=["POST"])
def download_report():
    auth_error = _require_api_login()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True) or {}
    patient_name = str(payload.get("patient_name", "Unknown")).strip() or "Unknown"
    prediction_result = str(payload.get("prediction_result", "N/A")).strip() or "N/A"
    confidence = float(payload.get("confidence", 0.0))
    risk_level = str(payload.get("risk_level", "N/A")).strip() or "N/A"
    image_path = str(payload.get("image_path", "N/A")).strip() or "N/A"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "Lung Scan Review - Prediction Report",
        f"Generated: {generated_at}",
        "",
        f"Patient Name: {patient_name}",
        f"Prediction: {prediction_result}",
        f"Confidence: {round(confidence * 100, 2)}%",
        f"Risk Level: {risk_level}",
        f"Image Path: {image_path}",
        "",
        "Disclaimer:",
        "This system is for research purposes only and not a replacement for medical diagnosis.",
    ]

    pdf = _build_simple_pdf(lines)
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{uuid.uuid4().hex[:8]}.pdf"'},
    )


_init_db()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug, host="0.0.0.0", port=port)
