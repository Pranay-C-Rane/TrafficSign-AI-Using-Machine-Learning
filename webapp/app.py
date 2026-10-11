
import json
import os
import time
import traceback

import cv2
import numpy as np
import onnxruntime as ort

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MAX_CAMERA_FRAME_SIZE = 5 * 1024 * 1024
YOLO_CONFIDENCE_THRESHOLD = 0.40
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

APP_FOLDER = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(os.path.join(APP_FOLDER, "model")):
    PROJECT_FOLDER = APP_FOLDER
else:
    PROJECT_FOLDER = os.path.dirname(APP_FOLDER)

MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "model")
UPLOAD_FOLDER = os.path.join(APP_FOLDER, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CNN_MODEL_PATH = os.path.join(MODEL_FOLDER, "traffic_sign_model.keras")
YOLO_MODEL_PATH = os.path.join(MODEL_FOLDER, "best.onnx")
CLASSES_PATH = os.path.join(MODEL_FOLDER, "classes.json")


# ============================================================
# 2. GTSRB CLASS NAMES
# ============================================================

GTSRB_NAMES = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for vehicles over 3.5 metric tons",
]


# ============================================================
# 3. LOAD CLASS MAPPING
# ============================================================

classes = {}

if os.path.isfile(CLASSES_PATH):
    try:
        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            classes = json.load(file)
        print("Class mapping loaded.")
    except (OSError, json.JSONDecodeError) as error:
        print("Could not read classes.json:", error)
else:
    print("classes.json not found.")


def mapped_class_name(class_id):
    """Return a real class name only when classes.json provides one."""
    class_id = int(class_id)

    if isinstance(classes, dict):
        value = classes.get(str(class_id), classes.get(class_id))
        if value is not None and str(value) != str(class_id):
            return str(value)

    if isinstance(classes, list) and 0 <= class_id < len(classes):
        value = classes[class_id]
        if str(value) != str(class_id):
            return str(value)

    return None


def get_gtsrb_class_name(class_id):
    """Name for the CNN's 43-class GTSRB output."""
    name = mapped_class_name(class_id)
    if name:
        return name

    if 0 <= int(class_id) < len(GTSRB_NAMES):
        return GTSRB_NAMES[int(class_id)]

    return f"Traffic Sign Class {class_id}"


def get_yolo_class_name(class_id):
    """Avoid incorrectly treating detector IDs as GTSRB IDs."""
    name = mapped_class_name(class_id)
    return name if name else f"Detected sign class {int(class_id)}"


# ============================================================
# 4. LOAD CNN MODEL (OPTIONAL)
# ============================================================

model = None
load_model = None

try:
    from tensorflow.keras.models import load_model as keras_load_model
    load_model = keras_load_model
except (ImportError, OSError) as error:
    print("TensorFlow unavailable:", error)

if load_model is not None and os.path.isfile(CNN_MODEL_PATH):
    try:
        print("Loading CNN model...")
        model = load_model(CNN_MODEL_PATH)
        print("CNN model loaded successfully.")
    except Exception:
        print("CNN model could not be loaded.")
        traceback.print_exc()
else:
    print("CNN unavailable. ONNX fallback will be used where possible.")


# ============================================================
# 5. LOAD YOLO ONNX MODEL
# ============================================================

yolo_session = None
yolo_input_name = None

if os.path.isfile(YOLO_MODEL_PATH):
    try:
        print("Loading YOLO ONNX model...")
        yolo_session = ort.InferenceSession(
            YOLO_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        yolo_input_name = yolo_session.get_inputs()[0].name

        print("YOLO ONNX model loaded.")
        print("YOLO input:", yolo_session.get_inputs()[0].shape)
        print("YOLO output:", yolo_session.get_outputs()[0].shape)
    except Exception:
        print("YOLO ONNX model could not be loaded.")
        traceback.print_exc()
else:
    print("YOLO model not found:", YOLO_MODEL_PATH)


# ============================================================
# 6. IMAGE HELPERS
# ============================================================

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def save_unique_image(image_file):
    safe_name = secure_filename(image_file.filename)

    if not safe_name:
        raise ValueError("Invalid image filename.")

    name, extension = os.path.splitext(safe_name)
    filename = safe_name
    counter = 1

    while os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
        filename = f"{name}_{counter}{extension}"
        counter += 1

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    return filename, image_path


def get_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ============================================================
# 7. CNN CLASSIFICATION
# ============================================================

def classify_with_cnn(image):
    if model is None:
        raise RuntimeError("CNN model is unavailable.")

    # Match the CNN's 32x32 RGB, normalized input.
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(
        rgb.astype(np.float32) / 255.0,
        axis=0,
    )

    prediction = np.asarray(model.predict(input_tensor, verbose=0)[0])

    if prediction.ndim != 1 or prediction.size == 0:
        raise ValueError("Unexpected CNN prediction output.")

    top_indices = prediction.argsort()[-3:][::-1]

    top_predictions = [
        {
            "name": get_gtsrb_class_name(int(class_id)),
            "confidence": round(float(prediction[class_id]) * 100, 2),
        }
        for class_id in top_indices
    ]

    predicted_class = int(top_indices[0])

    return {
        "class_id": predicted_class,
        "name": get_gtsrb_class_name(predicted_class),
        "confidence": round(float(prediction[predicted_class]) * 100, 2),
        "top_predictions": top_predictions,
        "method": "CNN",
    }


# ============================================================
# 8. YOLO DETECTION
# ============================================================

def detect_traffic_sign(image):
    if yolo_session is None:
        return None

    original_height, original_width = image.shape[:2]

    resized = cv2.resize(image, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor.astype(np.float32) / 255.0, axis=0)

    outputs = yolo_session.run(None, {yolo_input_name: tensor})

    if not outputs:
        return None

    predictions = np.asarray(outputs[0])

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.ndim != 2:
        return None

    # Common YOLO output: (features, candidates).
    if predictions.shape[0] < predictions.shape[1]:
        rows = predictions.T
    else:
        rows = predictions

    best = None
    best_confidence = YOLO_CONFIDENCE_THRESHOLD

    for row in rows:
        if len(row) < 5:
            continue

        scores = row[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if not np.isfinite(confidence) or confidence <= best_confidence:
            continue

        cx, cy, width, height = map(float, row[:4])

        best_confidence = confidence
        best = {
            "class_id": class_id,
            "confidence": confidence,
            "x1": cx - width / 2,
            "y1": cy - height / 2,
            "x2": cx + width / 2,
            "y2": cy + height / 2,
        }

    if best is None:
        return None

    # The source image was stretched to 640x640.
    sx = original_width / 640.0
    sy = original_height / 640.0

    x1 = max(0, min(int(best["x1"] * sx), original_width - 1))
    y1 = max(0, min(int(best["y1"] * sy), original_height - 1))
    x2 = max(0, min(int(best["x2"] * sx), original_width))
    y2 = max(0, min(int(best["y2"] * sy), original_height))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return {
        "class_id": best["class_id"],
        "confidence": best["confidence"],
        "box": [x1, y1, x2, y2],
        "crop": crop,
    }


# ============================================================
# 9. PREDICTION WITH FALLBACK
# ============================================================

def predict_image(image):
    # Prefer the 43-class CNN whenever it is available.
    if model is not None:
        try:
            return classify_with_cnn(image)
        except Exception:
            print("CNN prediction failed; trying ONNX detector.")
            traceback.print_exc()

    # The ONNX model is a detector, not a guaranteed 43-class classifier.
    detection = detect_traffic_sign(image)

    if detection is None:
        raise RuntimeError(
            "The CNN is unavailable and the ONNX detector did not find "
            "a sign. Check the server terminal and try a clear sign image."
        )

    name = get_yolo_class_name(detection["class_id"])
    confidence = round(float(detection["confidence"]) * 100, 2)

    return {
        "class_id": int(detection["class_id"]),
        "name": name,
        "confidence": confidence,
        "top_predictions": [
            {"name": name, "confidence": confidence}
        ],
        "method": "YOLO ONNX detector",
        "detection": detection,
    }


# ============================================================
# 10. SERVE UPLOADED IMAGES
# ============================================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ============================================================
# 11. HOME PAGE AND IMAGE UPLOAD
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    predicted_name = None
    confidence = None
    image_filename = None
    top_predictions = []
    error_message = None
    blur_warning = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if image_file is None or not image_file.filename:
            error_message = "Please select an image before prediction."

        elif not allowed_file(image_file.filename):
            error_message = (
                "Unsupported image format. Upload JPG, JPEG, PNG, BMP or WEBP."
            )

        else:
            try:
                image_filename, image_path = save_unique_image(image_file)
                image = cv2.imread(image_path)

                if image is None:
                    error_message = "Invalid image. Please upload another image."
                else:
                    if get_blur_score(image) < 50:
                        blur_warning = (
                            "The image may be blurry. Prediction could be less accurate."
                        )

                    result = predict_image(image)
                    predicted_name = result["name"]
                    confidence = result["confidence"]
                    top_predictions = result["top_predictions"]

            except Exception as error:
                print("Image prediction failed:", error)
                traceback.print_exc()
                error_message = str(error)

    return render_template(
        "index.html",
        predicted_name=predicted_name,
        confidence=confidence,
        image_filename=image_filename,
        top_predictions=top_predictions,
        error_message=error_message,
        blur_warning=blur_warning,
    )


# ============================================================
# 12. LIVE CAMERA API
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():
    try:
        image_file = request.files.get("image")

        if image_file is None:
            return jsonify({
                "detected": False,
                "message": "No camera frame received.",
            }), 400

        image_bytes = image_file.read(MAX_CAMERA_FRAME_SIZE + 1)

        if not image_bytes:
            return jsonify({
                "detected": False,
                "message": "Empty camera frame.",
            }), 400

        if len(image_bytes) > MAX_CAMERA_FRAME_SIZE:
            return jsonify({
                "detected": False,
                "message": "Camera frame is too large.",
            }), 413

        frame = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            return jsonify({
                "detected": False,
                "message": "Invalid camera frame.",
            }), 400

        detection = detect_traffic_sign(frame)

        if detection is None:
            return jsonify({
                "detected": False,
                "message": "No sign detected with sufficient confidence.",
            })

        crop_filename = f"live_sign_{time.time_ns()}.jpg"
        crop_path = os.path.join(UPLOAD_FOLDER, crop_filename)

        if not cv2.imwrite(crop_path, detection["crop"]):
            raise OSError("Could not save the detected sign crop.")

        # Classify the crop with the CNN if available.
        if model is not None:
            try:
                result = classify_with_cnn(detection["crop"])
                sign_name = result["name"]
                cnn_confidence = result["confidence"]
                method = "CNN"
            except Exception:
                sign_name = get_yolo_class_name(detection["class_id"])
                cnn_confidence = None
                method = "YOLO ONNX detector"
        else:
            sign_name = get_yolo_class_name(detection["class_id"])
            cnn_confidence = None
            method = "YOLO ONNX detector"

        return jsonify({
            "detected": True,
            "yolo_class_id": int(detection["class_id"]),
            "yolo_confidence": round(
                float(detection["confidence"]) * 100, 2
            ),
            "sign": sign_name,
            "cnn_confidence": cnn_confidence,
            "method": method,
            "box": detection["box"],
            "image": f"/uploads/{crop_filename}",
        })

    except Exception:
        traceback.print_exc()
        return jsonify({
            "detected": False,
            "message": "Detection failed. Check the server terminal.",
        }), 500


# ============================================================
# 13. HEALTH CHECK
# ============================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "cnn": "loaded" if model is not None else "unavailable",
        "yolo": "loaded" if yolo_session is not None else "unavailable",
        "classes": len(classes) if classes else len(GTSRB_NAMES),
    })


# ============================================================
# 14. ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        predicted_name=None,
        confidence=None,
        image_filename=None,
        top_predictions=[],
        error_message="Image is too large. Maximum allowed size is 10 MB.",
        blur_warning=None,
    ), 413


@app.errorhandler(500)
def internal_server_error(error):
    print("FLASK INTERNAL SERVER ERROR:", error)

    return render_template(
        "index.html",
        predicted_name=None,
        confidence=None,
        image_filename=None,
        top_predictions=[],
        error_message="Something went wrong on the server. Please try again.",
        blur_warning=None,
    ), 500


# ============================================================
# 15. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
