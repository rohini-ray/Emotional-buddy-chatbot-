import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "text_to_emotion_textcnn.keras"
)

TOKENIZER_PATH = os.path.join(
    MODELS_DIR,
    "tokenizer.json"
)

TEXT_LABELS = [
    "anger",
    "fear",
    "joy",
    "love",
    "neutral",
    "sadness",
    "surprise"
]

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(
        json.dumps(json.load(f))
    )

MAX_LEN = model.input_shape[1]


def predict_text_emotion(text):

    seq = tokenizer.texts_to_sequences([text])

    x = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probs = model.predict(
        x,
        verbose=0
    )[0]

    idx = np.argmax(probs)

    return {
        "emotion": TEXT_LABELS[idx],
        "confidence": float(probs[idx]),
        "all_probs": probs.tolist()
    }