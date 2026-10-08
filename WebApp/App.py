# ============================================================
# TRAFFIC SIGN AI - FLASK WEB APPLICATION
# CNN CLASSIFICATION + YOLO DETECTION
# ============================================================

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

import cv2
import json
import os
import time
import traceback

import numpy as np
import onnxruntime as ort


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Maximum uploaded image size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_FOLDER = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# IMPORTANT:
# Model folder is lowercase
MODEL_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "model"
)

# IMPORTANT:
# Upload folder is lowercase
UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. MODEL FILE PATHS
# ============================================================

CNN_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "traffic_sign_model.keras"
)

YOLO_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "best.onnx"
)

CLASSES_PATH = os.path.join(
    MODEL_FOLDER,
    "classes.json"
)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

required_files = [
    CNN_MODEL_PATH,
    YOLO_MODEL_PATH,
    CLASSES_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# 5. LOAD CNN MODEL
# ============================================================

print()
print("==============================================")
print("Loading CNN model...")
print("==============================================")

model = load_model(
    CNN_MODEL_PATH
)

print(
    "CNN model loaded successfully."
)


# ============================================================
# 6. LOAD YOLO MODEL
# ============================================================

print()
print("==============================================")
print("Loading YOLO model...")
print("==============================================")

yolo_session = ort.InferenceSession(
    YOLO_MODEL_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)

yolo_input_name = (
    yolo_session
    .get_inputs()[0]
    .name
)

print(
    "YOLO model loaded successfully."
)

print(
    "YOLO input:",
    yolo_input_name
)


# ============================================================
# 7. LOAD TRAFFIC SIGN CLASSES
# ============================================================

print()
print("==============================================")
print("Loading traffic sign classes...")
print("==============================================")

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as file:

    classes = json.load(file)


print(
    "Traffic sign classes loaded successfully."
)

print(
    "Number of classes:",
    len(classes)
)


# ============================================================
# 8. ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 9. CHECK FILE TYPE
# ============================================================

def allowed_file(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# 10. GET CLASS NAME
# ============================================================

def get_class_name(class_id):

    class_id = int(class_id)

    if isinstance(classes, dict):

        class_name = classes.get(
            str(class_id)
        )

        if class_name is not None:

            return str(
                class_name
            )

        class_name = classes.get(
            class_id
        )

        if class_name is not None:

            return str(
                class_name
            )

    if isinstance(classes, list):

        if 0 <= class_id < len(classes):

            return str(
                classes[class_id]
            )

    return f"Traffic Sign Class {class_id}"


# ============================================================
# 11. IMAGE SHARPNESS / BLUR
# ============================================================

def get_blur_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return float(
        sharpness
    )


def is_blurry(image):

    sharpness = get_blur_score(
        image
    )

    blur_threshold = 50

    return sharpness < blur_threshold


# ============================================================
# 12. TRAFFIC SIGN COLOR MASKS
# ============================================================

def get_color_masks(image):

    resized = cv2.resize(
        image,
        (300, 300),
        interpolation=cv2.INTER_AREA
    )

    hsv = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2HSV
    )

    red_1 = cv2.inRange(
        hsv,
        (0, 60, 40),
        (12, 255, 255)
    )

    red_2 = cv2.inRange(
        hsv,
        (168, 60, 40),
        (180, 255, 255)
    )

    red_mask = cv2.bitwise_or(
        red_1,
        red_2
    )

    blue_mask = cv2.inRange(
        hsv,
        (90, 50, 40),
        (140, 255, 255)
    )

    yellow_mask = cv2.inRange(
        hsv,
        (15, 50, 40),
        (40, 255, 255)
    )

    return (
        red_mask,
        blue_mask,
        yellow_mask
    )


# ============================================================
# 13. CHECK SIGN-LIKE SHAPE
# ============================================================

def has_sign_like_shape(image):

    resized = cv2.resize(
        image,
        (400, 400),
        interpolation=cv2.INTER_AREA
    )

    red_mask, blue_mask, yellow_mask = (
        get_color_masks(resized)
    )

    color_mask = cv2.bitwise_or(
        red_mask,
        blue_mask
    )

    color_mask = cv2.bitwise_or(
        color_mask,
        yellow_mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = (
        color_mask.shape[0]
        * color_mask.shape[1]
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 500:
            continue

        x, y, width, height = (
            cv2.boundingRect(
                contour
            )
        )

        if width < 20 or height < 20:
            continue

        rectangle_area = (
            width * height
        )

        if rectangle_area <= 0:
            continue

        region_ratio = (
            area / float(image_area)
        )

        if region_ratio < 0.003:
            continue

        aspect_ratio = (
            width / float(height)
        )

        if (
            aspect_ratio < 0.45
            or aspect_ratio > 2.2
        ):
            continue

        fill_ratio = (
            area
            / float(rectangle_area)
        )

        if fill_ratio >= 0.20:

            return True

    return False


# ============================================================
# 14. CHECK SIGN-LIKE BORDER
# ============================================================

def has_sign_like_border(image):

    resized = cv2.resize(
        image,
        (400, 400),
        interpolation=cv2.INTER_AREA
    )

    red_mask, blue_mask, yellow_mask = (
        get_color_masks(resized)
    )

    combined = cv2.bitwise_or(
        red_mask,
        blue_mask
    )

    combined = cv2.bitwise_or(
        combined,
        yellow_mask
    )

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 700:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        circularity = (
            4 * np.pi * area
        ) / (
            perimeter * perimeter
        )

        x, y, width, height = (
            cv2.boundingRect(
                contour
            )
        )

        if width < 25 or height < 25:
            continue

        aspect_ratio = (
            width / float(height)
        )

        reasonable_ratio = (
            0.55
            <= aspect_ratio
            <= 1.80
        )

        if (
            circularity >= 0.35
            and reasonable_ratio
        ):

            return True

    return False


# ============================================================
# 15. TRAFFIC SIGN VISUAL CHECK
# ============================================================

def looks_like_traffic_sign(image):

    shape_check = (
        has_sign_like_shape(image)
    )

    border_check = (
        has_sign_like_border(image)
    )

    print(
        "Shape Check:",
        shape_check
    )

    print(
        "Border Check:",
        border_check
    )

    return (
        shape_check
        or border_check
    )


# ============================================================
# 16. YOLO TRAFFIC SIGN DETECTION
# ============================================================

def detect_traffic_sign(image):

    original_height, original_width = (
        image.shape[:2]
    )

    resized = cv2.resize(
        image,
        (640, 640),
        interpolation=cv2.INTER_LINEAR
    )

    resized = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    resized = np.transpose(
        resized,
        (2, 0, 1)
    )

    resized = resized.astype(
        np.float32
    ) / 255.0

    input_tensor = np.expand_dims(
        resized,
        axis=0
    )

    outputs = yolo_session.run(
        None,
        {
            yolo_input_name:
                input_tensor
        }
    )

    if not outputs:

        return None

    predictions = outputs[0]

    if predictions.ndim == 3:

        predictions = predictions[0]

    if predictions.ndim != 2:

        print(
            "Unexpected YOLO output shape:",
            predictions.shape
        )

        return None

    if predictions.shape[0] < predictions.shape[1]:

        detection_rows = predictions.T

    else:

        detection_rows = predictions

    best_detection = None

    best_confidence = 0.0

    for prediction in detection_rows:

        if len(prediction) < 5:

            continue

        center_x = float(
            prediction[0]
        )

        center_y = float(
            prediction[1]
        )

        box_width = float(
            prediction[2]
        )

        box_height = float(
            prediction[3]
        )

        class_scores = prediction[4:]

        if len(class_scores) == 0:

            continue

        class_id = int(
            np.argmax(
                class_scores
            )
        )

        confidence = float(
            class_scores[class_id]
        )

        if confidence < 0.40:

            continue

        if confidence > best_confidence:

            best_confidence = confidence

            x1 = int(
                center_x
                - box_width / 2
            )

            y1 = int(
                center_y
                - box_height / 2
            )

            x2 = int(
                center_x
                + box_width / 2
            )

            y2 = int(
                center_y
                + box_height / 2
            )

            best_detection = {

                "class_id":
                    class_id,

                "confidence":
                    confidence,

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2
            }

    if best_detection is None:

        return None

    scale_x = (
        original_width
        / 640.0
    )

    scale_y = (
        original_height
        / 640.0
    )

    x1 = int(
        best_detection["x1"]
        * scale_x
    )

    y1 = int(
        best_detection["y1"]
        * scale_y
    )

    x2 = int(
        best_detection["x2"]
        * scale_x
    )

    y2 = int(
        best_detection["y2"]
        * scale_y
    )

    x1 = max(
        0,
        min(
            x1,
            original_width - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            original_height - 1
        )
    )

    x2 = max(
        0,
        min(
            x2,
            original_width
        )
    )

    y2 = max(
        0,
        min(
            y2,
            original_height
        )
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):

        return None

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return None

    return {

        "class_id":
            best_detection["class_id"],

        "confidence":
            best_detection["confidence"],

        "box": [
            x1,
            y1,
            x2,
            y2
        ],

        "crop":
            crop
    }


# ============================================================
# 17. CNN CLASSIFICATION
# ============================================================

def classify_image(image):

    resized = cv2.resize(
        image,
        (32, 32),
        interpolation=cv2.INTER_AREA
    )

    resized = resized.astype(
        "float32"
    ) / 255.0

    resized = resized.reshape(
        1,
        32,
        32,
        3
    )

    prediction = model.predict(
        resized,
        verbose=0
    )[0]

    top_3_indices = (
        prediction.argsort()[
            -3:
        ][::-1]
    )

    top_predictions = []

    for class_id in top_3_indices:

        class_id = int(
            class_id
        )

        class_name = (
            get_class_name(
                class_id
            )
        )

        class_confidence = (
            float(
                prediction[
                    class_id
                ]
                * 100
            )
        )

        top_predictions.append({

            "name":
                class_name,

            "confidence":
                round(
                    class_confidence,
                    2
                )
        })

    predicted_class = int(
        top_3_indices[0]
    )

    predicted_name = (
        get_class_name(
            predicted_class
        )
    )

    confidence = float(
        prediction[
            predicted_class
        ]
        * 100
    )

    return {

        "class_id":
            predicted_class,

        "name":
            predicted_name,

        "confidence":
            round(
                confidence,
                2
            ),

        "top_predictions":
            top_predictions
    }


# ============================================================
# 18. SERVE UPLOADED IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# 19. HOME + IMAGE PREDICTION
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def home():

    predicted_name = None
    confidence = None
    image_filename = None
    top_predictions = []
    error_message = None
    blur_warning = None

    try:

        if request.method == "POST":

            image = request.files.get(
                "image"
            )

            if image is None:

                error_message = (
                    "Please select an image before prediction."
                )

            elif image.filename == "":

                error_message = (
                    "Please select an image before prediction."
                )

            elif not allowed_file(
                image.filename
            ):

                error_message = (
                    "Unsupported image format. "
                    "Please upload JPG, JPEG, PNG, BMP or WEBP."
                )

            else:

                filename = secure_filename(
                    image.filename
                )

                name, extension = (
                    os.path.splitext(
                        filename
                    )
                )

                counter = 1

                original_filename = filename

                while os.path.exists(
                    os.path.join(
                        UPLOAD_FOLDER,
                        filename
                    )
                ):

                    filename = (
                        f"{name}_{counter}{extension}"
                    )

                    counter += 1

                image_path = os.path.join(
                    UPLOAD_FOLDER,
                    filename
                )

                image.save(
                    image_path
                )

                image_filename = filename

                img = cv2.imread(
                    image_path
                )

                if img is None:

                    error_message = (
                        "The selected image cannot be processed. "
                        "Please upload a valid image."
                    )

                else:

                    height, width = (
                        img.shape[:2]
                    )

                    print()
                    print(
                        "=============================================="
                    )

                    print(
                        "Uploaded Image:",
                        original_filename
                    )

                    print(
                        "Image Size:",
                        width,
                        "x",
                        height
                    )

                    blur_score = (
                        get_blur_score(
                            img
                        )
                    )

                    print(
                        "Blur / Sharpness Score:",
                        round(
                            blur_score,
                            2
                        )
                    )

                    if is_blurry(img):

                        blur_warning = (
                            "The image appears blurry, "
                            "but the CNN will still analyze it."
                        )

                        print(
                            "Image Quality: BLURRY"
                        )

                    else:

                        print(
                            "Image Quality: CLEAR"
                        )

                    sign_like = (
                        looks_like_traffic_sign(
                            img
                        )
                    )

                    print(
                        "Traffic Sign Visual Check:",
                        sign_like
                    )

                    result = classify_image(
                        img
                    )

                    predicted_name = (
                        result["name"]
                    )

                    confidence = (
                        result["confidence"]
                    )

                    top_predictions = (
                        result["top_predictions"]
                    )

                    print(
                        "Predicted Class ID:",
                        result["class_id"]
                    )

                    print(
                        "Predicted Traffic Sign:",
                        predicted_name
                    )

                    print(
                        "Confidence:",
                        confidence,
                        "%"
                    )

                    print(
                        "Top 3 Predictions:"
                    )

                    for index, item in enumerate(
                        top_predictions,
                        start=1
                    ):

                        print(
                            index,
                            ".",
                            item["name"],
                            "-",
                            item["confidence"],
                            "%"
                        )

                    print(
                        "=============================================="
                    )

    except Exception as error:

        print()
        print(
            "=============================================="
        )

        print(
            "ERROR DURING IMAGE PROCESSING"
        )

        print(
            str(error)
        )

        traceback.print_exc()

        print(
            "=============================================="
        )

        error_message = (
            "An error occurred while processing the image. "
            "Please try another image."
        )

    return render_template(

        "index.html",

        predicted_name=predicted_name,

        confidence=confidence,

        image_filename=image_filename,

        top_predictions=top_predictions,

        error_message=error_message,

        blur_warning=blur_warning

    )


# ============================================================
# 20. LIVE CAMERA DETECTION API
# ============================================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    try:

        if "image" not in request.files:

            return {

                "detected":
                    False,

                "message":
                    "No camera frame received."
            }

        image_file = (
            request.files["image"]
        )

        image_bytes = (
            image_file.read()
        )

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return {

                "detected":
                    False,

                "message":
                    "Invalid camera frame."
            }

        detection = (
            detect_traffic_sign(
                frame
            )
        )

        if detection is None:

            return {

                "detected":
                    False,

                "message":
                    "No traffic sign detected."
            }

        crop = detection[
            "crop"
        ]

        crop_filename = (
            f"live_sign_"
            f"{int(time.time() * 1000)}.jpg"
        )

        crop_path = os.path.join(
            UPLOAD_FOLDER,
            crop_filename
        )

        cv2.imwrite(
            crop_path,
            crop
        )

        yolo_class_id = int(
            detection[
                "class_id"
            ]
        )

        yolo_confidence = float(
            detection[
                "confidence"
            ]
        )

        result = classify_image(
            crop
        )

        predicted_name = (
            result["name"]
        )

        cnn_confidence = (
            result["confidence"]
        )

        x1, y1, x2, y2 = (
            detection["box"]
        )

        return {

            "detected":
                True,

            "yolo_class_id":
                yolo_class_id,

            "yolo_confidence":
                round(
                    yolo_confidence * 100,
                    2
                ),

            "sign":
                predicted_name,

            "cnn_confidence":
                cnn_confidence,

            "box": [
                x1,
                y1,
                x2,
                y2
            ],

            "image":
                f"/uploads/{crop_filename}"
        }

    except Exception as error:

        print()
        print(
            "=============================================="
        )

        print(
            "LIVE DETECTION ERROR"
        )

        print(
            str(error)
        )

        traceback.print_exc()

        print(
            "=============================================="
        )

        return {

            "detected":
                False,

            "message":
                "Detection error occurred."
        }, 500


# ============================================================
# 21. HEALTH CHECK
# ============================================================

@app.route(
    "/health"
)
def health():

    return {

        "status":
            "ok",

        "cnn":
            "loaded",

        "yolo":
            "loaded",

        "classes":
            len(classes)
    }


# ============================================================
# 22. ERROR HANDLERS
# ============================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    return render_template(

        "index.html",

        predicted_name=None,

        confidence=None,

        image_filename=None,

        top_predictions=[],

        error_message=(
            "Image is too large. "
            "Maximum allowed size is 10 MB."
        ),

        blur_warning=None

    ), 413


@app.errorhandler(
    500
)
def internal_server_error(error):

    print()
    print(
        "=============================================="
    )

    print(
        "FLASK INTERNAL SERVER ERROR"
    )

    print(
        str(error)
    )

    traceback.print_exc()

    print(
        "=============================================="
    )

    return render_template(

        "index.html",

        predicted_name=None,

        confidence=None,

        image_filename=None,

        top_predictions=[],

        error_message=(
            "Something went wrong on the server. "
            "Please try again."
        ),

        blur_warning=None

    ), 500


# ============================================================
# 23. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )