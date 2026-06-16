import os
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "facial_emotion_model.keras"
)

FACE_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

model = tf.keras.models.load_model(MODEL_PATH)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def predict_face_emotion(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Cannot read image")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        1.3,
        5
    )

    if len(faces) == 0:
        return {
            "emotion": "neutral",
            "confidence": 0.0
        }

    x, y, w, h = faces[0]

    roi = gray[y:y+h, x:x+w]

    roi = cv2.resize(
        roi,
        (48, 48)
    )

    roi = roi.astype("float32") / 255.0

    roi = np.expand_dims(
        roi,
        axis=(0, -1)
    )

    probs = model.predict(
        roi,
        verbose=0
    )[0]

    idx = np.argmax(probs)

    return {
        "emotion": FACE_LABELS[idx],
        "confidence": float(probs[idx]),
        "all_probs": probs.tolist()
    }