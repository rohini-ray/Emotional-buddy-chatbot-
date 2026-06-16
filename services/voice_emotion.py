import os
import librosa
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "voice_emotion_model.h5"
)

VOICE_LABELS = [
    "disgust",
    "angry",
    "fear",
    "sad",
    "happy",
    "neutral",
    "pleasant",
    "surprise",
    "calm"
]

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

TARGET_SR = 16000
TARGET_LEN = 526


def extract_features(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=40
    )

    features = mfcc.T.reshape(-1)

    if len(features) < TARGET_LEN:
        features = np.pad(
            features,
            (0, TARGET_LEN - len(features))
        )
    else:
        features = features[:TARGET_LEN]

    return features.reshape(
        1,
        TARGET_LEN,
        1
    )


def predict_voice_emotion(audio_path):

    x = extract_features(audio_path)

    probs = model.predict(
        x,
        verbose=0
    )[0]

    idx = np.argmax(probs)

    return {
        "emotion": VOICE_LABELS[idx],
        "confidence": float(probs[idx]),
        "all_probs": probs.tolist()
    }