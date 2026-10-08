# ================================
# 1. IMPORT LIBRARIES
# ================================

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

import cv2
import json
import os
import time
import numpy as np
import onnxruntime as ort


# ================================
# 2. CREATE FLASK APPLICATION
# ================================

app = Flask(__name__)


# ================================
# 3. PROJECT PATHS
# ================================

PROJECT_FOLDER = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "Model"
)

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "Uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ================================
# 4. LOAD CNN MODEL
# ================================

model_path = os.path.join(
    MODEL_FOLDER,
    "traffic_sign_model.keras"
)

model = load_model(
    model_path
)


# ================================
# 5. LOAD YOLO DETECTION MODEL
# ================================

yolo_model_path = os.path.join(
    MODEL_FOLDER,
    "best.onnx"
)

yolo_session = ort.InferenceSession(
    yolo_model_path,
    providers=["CPUExecutionProvider"]
)

yolo_input_name = yolo_session.get_inputs()[0].name

print(
    "YOLO model loaded successfully."
)


# ================================
# 6. LOAD TRAFFIC SIGN CLASSES
# ================================

classes_path = os.path.join(
    MODEL_FOLDER,
    "Classes.json"
)

with open(
    classes_path,
    "r"
) as file:

    classes = json.load(file)


# ================================
# 7. ALLOWED IMAGE TYPES
# ================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ================================
# 8. CHECK FILE TYPE
# ================================

def allowed_file(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ================================
# 9. IMAGE SHARPNESS / BLUR
# ================================

def get_blur_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return sharpness


def is_blurry(image):

    sharpness = get_blur_score(
        image
    )

    blur_threshold = 50

    return sharpness < blur_threshold


# ================================
# 10. TRAFFIC SIGN VISUAL ANALYSIS
# ================================

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

    # ============================
    # RED MASK
    # ============================

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

    # ============================
    # BLUE MASK
    # ============================

    blue_mask = cv2.inRange(
        hsv,
        (90, 50, 40),
        (140, 255, 255)
    )

    # ============================
    # YELLOW MASK
    # ============================

    yellow_mask = cv2.inRange(
        hsv,
        (15, 50, 40),
        (40, 255, 255)
    )

    return red_mask, blue_mask, yellow_mask


# ================================
# 11. CHECK TRAFFIC SIGN SHAPE
# ================================

def has_sign_like_shape(image):

    resized = cv2.resize(
        image,
        (400, 400),
        interpolation=cv2.INTER_AREA
    )

    red_mask, blue_mask, yellow_mask = get_color_masks(
        resized
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

        x, y, width, height = cv2.boundingRect(
            contour
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
            area / float(rectangle_area)
        )

        if fill_ratio >= 0.20:

            return True

    return False


# ================================
# 12. CHECK TRAFFIC SIGN BORDER
# ================================

def has_sign_like_border(image):

    resized = cv2.resize(
        image,
        (400, 400),
        interpolation=cv2.INTER_AREA
    )

    red_mask, blue_mask, yellow_mask = get_color_masks(
        resized
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

        x, y, width, height = cv2.boundingRect(
            contour
        )

        if width < 25 or height < 25:
            continue

        aspect_ratio = (
            width / float(height)
        )

        reasonable_ratio = (
            0.55 <= aspect_ratio <= 1.80
        )

        if (
            circularity >= 0.35
            and reasonable_ratio
        ):

            return True

    return False


# ================================
# 13. TRAFFIC SIGN VISUAL CHECK
# ================================

def looks_like_traffic_sign(image):

    shape_check = has_sign_like_shape(
        image
    )

    border_check = has_sign_like_border(
        image
    )

    print(
        "Shape Check:",
        shape_check
    )

    print(
        "Border Check:",
        border_check
    )

    if shape_check or border_check:

        return True

    return False


# ================================
# 14. YOLO TRAFFIC SIGN DETECTION
# ================================

def detect_traffic_sign(image):

    original_height, original_width = image.shape[:2]

    # ============================
    # RESIZE TO YOLO INPUT
    # ============================

    resized = cv2.resize(
        image,
        (640, 640),
        interpolation=cv2.INTER_LINEAR
    )

    # ============================
    # BGR → RGB
    # ============================

    resized = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # ============================
    # HWC → CHW
    # ============================

    resized = np.transpose(
        resized,
        (2, 0, 1)
    )

    # ============================
    # NORMALIZE
    # ============================

    resized = resized.astype(
        np.float32
    ) / 255.0

    # ============================
    # ADD BATCH DIMENSION
    # ============================

    input_tensor = np.expand_dims(
        resized,
        axis=0
    )

    # ============================
    # YOLO INFERENCE
    # ============================

    outputs = yolo_session.run(
        None,
        {
            yolo_input_name: input_tensor
        }
    )

    predictions = outputs[0][0]

    best_detection = None
    best_confidence = 0.0

    # ============================
    # PROCESS PREDICTIONS
    # ============================

    for prediction in predictions.T:

        center_x = prediction[0]
        center_y = prediction[1]

        box_width = prediction[2]
        box_height = prediction[3]

        class_scores = prediction[4:]

        class_id = int(
            np.argmax(
                class_scores
            )
        )

        confidence = float(
            class_scores[class_id]
        )

        # Detection threshold

        if confidence < 0.40:
            continue

        # Keep strongest detection

        if confidence > best_confidence:

            best_confidence = confidence

            # Center → corners

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

                "class_id": class_id,

                "confidence": confidence,

                "x1": x1,

                "y1": y1,

                "x2": x2,

                "y2": y2

            }

    # ============================
    # NO DETECTION
    # ============================

    if best_detection is None:

        return None

    # ============================
    # SCALE BOX TO ORIGINAL IMAGE
    # ============================

    scale_x = (
        original_width / 640.0
    )

    scale_y = (
        original_height / 640.0
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

    # ============================
    # KEEP BOX INSIDE IMAGE
    # ============================

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

    # ============================
    # INVALID BOX
    # ============================

    if x2 <= x1 or y2 <= y1:

        return None

    # ============================
    # CROP DETECTED SIGN
    # ============================

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return None

    # ============================
    # RETURN DETECTION
    # ============================

    return {

        "class_id": best_detection["class_id"],

        "confidence": best_detection["confidence"],

        "box": [
            x1,
            y1,
            x2,
            y2
        ],

        "crop": crop

    }


# ================================
# 15. SERVE UPLOADED IMAGES
# ================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ================================
# 16. HOME + PREDICTION
# ================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    predicted_name = None

    confidence = None

    image_filename = None

    top_predictions = []

    error_message = None

    blur_warning = None

    # ============================
    # CHECK IMAGE SUBMISSION
    # ============================

    if request.method == "POST":

        image = request.files.get(
            "image"
        )

        # ============================
        # NO IMAGE
        # ============================

        if image is None:

            error_message = (
                "Please select an image before prediction."
            )

        elif image.filename == "":

            error_message = (
                "Please select an image before prediction."
            )

        # ============================
        # INVALID FILE TYPE
        # ============================

        elif not allowed_file(
            image.filename
        ):

            error_message = (
                "Unsupported image format. "
                "Please upload a JPG, JPEG, PNG, BMP or WEBP image."
            )

        else:

            # ============================
            # SAFE FILENAME
            # ============================

            filename = secure_filename(
                image.filename
            )

            # ============================
            # CREATE UNIQUE FILENAME
            # ============================

            name, extension = os.path.splitext(
                filename
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

            # ============================
            # SAVE IMAGE
            # ============================

            image_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            image.save(
                image_path
            )

            image_filename = filename

            # ============================
            # READ IMAGE
            # ============================

            img = cv2.imread(
                image_path
            )

            # ============================
            # UNREADABLE IMAGE
            # ============================

            if img is None:

                error_message = (
                    "The selected image cannot be processed. "
                    "Please upload a valid image."
                )

            else:

                # ============================
                # IMAGE DIMENSIONS
                # ============================

                height, width = img.shape[:2]

                print(
                    "\n=============================="
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

                # ============================
                # BLUR ANALYSIS
                # ============================

                blur_score = get_blur_score(
                    img
                )

                print(
                    "Blur / Sharpness Score:",
                    round(
                        blur_score,
                        2
                    )
                )

                # ============================
                # BLUR WARNING
                # ============================

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

                # ============================
                # TRAFFIC SIGN VISUAL CHECK
                # ============================

                sign_like = looks_like_traffic_sign(
                    img
                )

                print(
                    "Traffic Sign Visual Check:",
                    sign_like
                )

                # ============================
                # INVALID RANDOM IMAGE
                # ============================

                if not sign_like:

                    error_message = (
                        "Invalid image. "
                        "Please upload an image containing "
                        "a traffic sign."
                    )

                    print(
                        "Result: INVALID TRAFFIC SIGN IMAGE"
                    )

                # ============================
                # TRAFFIC SIGN → CNN
                # ============================

                else:

                    # ============================
                    # RESIZE IMAGE
                    # ============================

                    img = cv2.resize(
                        img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA
                    )

                    # ============================
                    # NORMALIZE
                    # ============================

                    img = img.astype(
                        "float32"
                    ) / 255.0

                    # ============================
                    # ADD BATCH DIMENSION
                    # ============================

                    img = img.reshape(
                        1,
                        32,
                        32,
                        3
                    )

                    # ============================
                    # CNN PREDICTION
                    # ============================

                    prediction = model.predict(
                        img,
                        verbose=0
                    )[0]

                    # ============================
                    # TOP 3
                    # ============================

                    top_3_indices = (
                        prediction.argsort()[
                            -3:
                        ][::-1]
                    )

                    for class_id in top_3_indices:

                        class_id = int(
                            class_id
                        )

                        class_name = classes[
                            str(class_id)
                        ]

                        class_confidence = float(
                            prediction[
                                class_id
                            ] * 100
                        )

                        top_predictions.append({

                            "name": class_name,

                            "confidence": round(
                                class_confidence,
                                2
                            )

                        })

                    # ============================
                    # MAIN PREDICTION
                    # ============================

                    predicted_class = int(
                        top_3_indices[0]
                    )

                    predicted_name = classes[
                        str(predicted_class)
                    ]

                    confidence = float(
                        prediction[
                            predicted_class
                        ] * 100
                    )

                    # ============================
                    # TERMINAL OUTPUT
                    # ============================

                    print(
                        "Predicted Class ID:",
                        predicted_class
                    )

                    print(
                        "Predicted Traffic Sign:",
                        predicted_name
                    )

                    print(
                        "Confidence:",
                        round(
                            confidence,
                            2
                        ),
                        "%"
                    )

                    print(
                        "Top 3 Predictions:"
                    )

                    for index, result in enumerate(
                        top_predictions,
                        start=1
                    ):

                        print(
                            index,
                            ".",
                            result["name"],
                            "-",
                            result["confidence"],
                            "%"
                        )

                    print(
                        "==============================\n"
                    )

    # ================================
    # SEND DATA TO HTML
    # ================================

    return render_template(

        "Index.html",

        predicted_name=predicted_name,

        confidence=(
            round(
                confidence,
                2
            )
            if confidence is not None
            else None
        ),

        image_filename=image_filename,

        top_predictions=top_predictions,

        error_message=error_message,

        blur_warning=blur_warning

    )


# ================================
# 17. LIVE CAMERA DETECTION API
# ================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    # ============================
    # CHECK FRAME
    # ============================

    if "image" not in request.files:

        return {
            "detected": False,
            "message": "No camera frame received."
        }

    image_file = request.files["image"]

    # ============================
    # READ IMAGE BYTES
    # ============================

    image_bytes = image_file.read()

    # ============================
    # CONVERT TO NUMPY ARRAY
    # ============================

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    # ============================
    # DECODE IMAGE
    # ============================

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    # ============================
    # CHECK IMAGE
    # ============================

    if frame is None:

        return {
            "detected": False,
            "message": "Invalid camera frame."
        }

    # ============================
    # YOLO DETECTION
    # ============================

    detection = detect_traffic_sign(
        frame
    )

    # ============================
    # NO TRAFFIC SIGN
    # ============================

    if detection is None:

        return {
            "detected": False,
            "message": "No traffic sign detected."
        }

    # ============================
    # GET DETECTED CROP
    # ============================

    crop = detection["crop"]

    # ============================
    # SAVE DETECTED CROP
    # ============================

    crop_filename = (
        f"live_sign_{int(time.time() * 1000)}.jpg"
    )

    crop_path = os.path.join(
        UPLOAD_FOLDER,
        crop_filename
    )

    cv2.imwrite(
        crop_path,
        crop
    )

    # ============================
    # YOLO INFORMATION
    # ============================

    yolo_class_id = int(
        detection["class_id"]
    )

    yolo_confidence = float(
        detection["confidence"]
    )

    # ============================
    # PREPARE CROP FOR CNN
    # ============================

    crop = cv2.resize(
        crop,
        (32, 32),
        interpolation=cv2.INTER_AREA
    )

    crop = crop.astype(
        "float32"
    ) / 255.0

    crop = crop.reshape(
        1,
        32,
        32,
        3
    )

    # ============================
    # CNN PREDICTION
    # ============================

    prediction = model.predict(
        crop,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(
            prediction
        )
    )

    predicted_name = classes[
        str(predicted_class)
    ]

    cnn_confidence = float(
        prediction[
            predicted_class
        ] * 100
    )

    # ============================
    # RETURN RESULT TO JAVASCRIPT
    # ============================

    x1, y1, x2, y2 = detection["box"]

    return {

        "detected": True,

        "yolo_class_id": yolo_class_id,

        "yolo_confidence": round(
            yolo_confidence * 100,
            2
        ),

        "sign": predicted_name,

        "cnn_confidence": round(
            cnn_confidence,
            2
        ),

        "box": [
            x1,
            y1,
            x2,
            y2
        ],

        "image": (
            f"/uploads/{crop_filename}"
        )

    }


# ================================
# RUN FLASK APPLICATION
# ================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )