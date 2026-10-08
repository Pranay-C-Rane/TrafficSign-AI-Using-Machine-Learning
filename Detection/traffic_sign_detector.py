import cv2
import numpy as np
import onnxruntime as ort
import json
from tensorflow.keras.models import load_model


# ==============================
# 1. File Paths
# ==============================

YOLO_MODEL_PATH = "../Model/best.onnx"
CNN_MODEL_PATH = "../Model/traffic_sign_model.keras"
CLASSES_PATH = "../Model/Classes.json"
IMAGE_PATH = "test_scene.jpg"


# ==============================
# 2. Load YOLO Detector
# ==============================

detector = ort.InferenceSession(YOLO_MODEL_PATH)

input_name = detector.get_inputs()[0].name


# ==============================
# 3. Load CNN Classifier
# ==============================

cnn_model = load_model(CNN_MODEL_PATH)

with open(CLASSES_PATH, "r") as file:
    class_names = json.load(file)


# ==============================
# 4. Read Image
# ==============================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Test image not found!")
    exit()


original_image = image.copy()

height, width = image.shape[:2]


# ==============================
# 5. YOLO Letterbox Preprocessing
# ==============================

scale = min(640 / width, 640 / height)

new_width = int(width * scale)
new_height = int(height * scale)

resized = cv2.resize(image, (new_width, new_height))

canvas = np.full((640, 640, 3), 114, dtype=np.uint8)

pad_x = (640 - new_width) // 2
pad_y = (640 - new_height) // 2

canvas[
    pad_y:pad_y + new_height,
    pad_x:pad_x + new_width
] = resized


# Convert BGR → RGB
input_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# Normalize pixels
input_image = input_image.astype(np.float32) / 255.0

# HWC → CHW
input_image = np.transpose(input_image, (2, 0, 1))

# Add batch dimension
input_image = np.expand_dims(input_image, axis=0)


# ==============================
# 6. YOLO Detection
# ==============================

outputs = detector.run(
    None,
    {input_name: input_image}
)

detections = outputs[0][0].T


confidence_threshold = 0.25

boxes = []
scores = []
class_ids = []


for detection in detections:

    x, y, box_width, box_height = detection[:4]

    class_scores = detection[4:]

    class_id = int(np.argmax(class_scores))

    confidence = float(class_scores[class_id])

    if confidence < confidence_threshold:
        continue

    # Convert center coordinates to corners
    x1 = x - box_width / 2
    y1 = y - box_height / 2
    x2 = x + box_width / 2
    y2 = y + box_height / 2

    # Remove letterbox padding
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # Keep coordinates inside image
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    boxes.append([x1, y1, x2 - x1, y2 - y1])
    scores.append(confidence)
    class_ids.append(class_id)


# ==============================
# 7. Remove Duplicate Boxes
# ==============================

indices = cv2.dnn.NMSBoxes(
    boxes,
    scores,
    confidence_threshold,
    0.45
)


# ==============================
# 8. Crop Sign + CNN Prediction
# ==============================

yolo_class_names = [
    "Bus stop",
    "Crossroad",
    "No entry",
    "No parking",
    "No stopping",
    "Speed limit",
    "Yield",
    "Direction of road"
]


if len(indices) == 0:

    print("No traffic sign detected.")

else:

    print(f"Detected {len(indices)} traffic sign(s):")

    for index in indices:

        index = int(index)

        x, y, w, h = boxes[index]

        x2 = x + w
        y2 = y + h

        # Crop detected traffic sign
        cropped_sign = original_image[y:y2, x:x2]

        if cropped_sign.size == 0:
            continue

        # Save cropped sign
        crop_path = f"detected_sign_{index}.jpg"

        cv2.imwrite(
            crop_path,
            cropped_sign
        )


        # ==============================
        # 9. Prepare Crop for CNN
        # ==============================

        cnn_image = cv2.resize(
            cropped_sign,
            (32, 32)
        )

        cnn_image = cv2.cvtColor(
            cnn_image,
            cv2.COLOR_BGR2RGB
        )

        cnn_image = cnn_image.astype(
            np.float32
        ) / 255.0

        cnn_image = np.expand_dims(
            cnn_image,
            axis=0
        )


        # ==============================
        # 10. CNN Prediction
        # ==============================

        predictions = cnn_model.predict(
            cnn_image,
            verbose=0
        )[0]

        cnn_class_id = int(
            np.argmax(predictions)
        )

        cnn_confidence = float(
            predictions[cnn_class_id]
        )


        # Get CNN class name
        cnn_class_name = str(
            class_names[str(cnn_class_id)]
            if isinstance(class_names, dict)
            else class_names[cnn_class_id]
        )


        print(
            f"YOLO: {yolo_class_names[class_ids[index]]} "
            f"| CNN: {cnn_class_name} "
            f"| Confidence: {cnn_confidence * 100:.2f}%"
        )


        # ==============================
        # 11. Draw Bounding Box
        # ==============================

        cv2.rectangle(
            original_image,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = (
            f"{cnn_class_name} "
            f"{cnn_confidence * 100:.1f}%"
        )

        cv2.putText(
            original_image,
            label,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


# ==============================
# 12. Save Final Result
# ==============================

cv2.imwrite(
    "detection_result.jpg",
    original_image
)

print("Detection result saved as: detection_result.jpg")