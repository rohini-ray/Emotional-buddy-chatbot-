import os
import re
import time
import uuid
import json
import tempfile
import traceback
import subprocess
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import librosa
import tensorflow as tf
import requests
import cv2

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ✅ Use PUBLIC TF-Keras APIs (not keras.src.*)
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import google.generativeai as genai
import os

LLM_API_KEY = "AQ.Ab8RN6JtANw-bucw6xunyLHVWy4TpMHWaLuuylsmuzkM0av1hg" 
print("LLM_API_KEY =", os.getenv("LLM_API_KEY"))
print("Loaded key =", LLM_API_KEY)
genai.configure(api_key=LLM_API_KEY)

chat_model = genai.GenerativeModel("gemini-2.5-flash")


# ----------------------------
# Strict env helpers (NO defaults)
# ----------------------------
def must_env(key: str) -> str:
    v = os.getenv(key, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {key}")
    return v

def opt_env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ----------------------------
# Generic LLM Config (anonymous)
# ----------------------------
# Required
LLM_PROVIDER = must_env("LLM_PROVIDER").lower()   # "anthropic" or "openai_compat"
LLM_API_KEY = must_env("LLM_API_KEY")
LLM_MODEL = must_env("LLM_MODEL")
LLM_API_URL = must_env("LLM_API_URL")

# Optional
LLM_API_VERSION = opt_env("LLM_API_VERSION", "")         # used by Anthropic
LLM_TIMEOUT_S = float(opt_env("LLM_TIMEOUT_S", "60"))

PORT = int(opt_env("PORT", "5000"))


# ----------------------------
# Local model config
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_to_emotion_textcnn.keras")
TOKENIZER_PATH  = os.path.join(MODELS_DIR, "tokenizer.json")
VOICE_MODEL_PATH = os.path.join(MODELS_DIR, "voice_emotion_model.h5")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "facial_emotion_model.keras")

TEXT_LABELS = ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
VOICE_LABELS = ["disgust", "angry", "fear", "sad", "happy", "neutral", "pleasant", "surprise", "calm"]
FACE_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

FACE_TO_STEER = {
    "Angry": "anger",
    "Disgust": "anger",
    "Fear": "anxiety",
    "Happy": "joy",
    "Sad": "sadness",
    "Surprise": "mixed",
    "Neutral": "neutral"
}


VOICE_TARGET_LEN = 526
TARGET_SR = 16000
N_MFCC = 40

FACE_IMAGE_SIZE = 48  # FER2013 standard size (will auto-adjust if model requires different)

LOW_CONF = 0.35
HIGH_CONF_SUSPICIOUS = 0.85

LONG_CHARS = 700
LONG_TOKENS_ROUGH = 250


# Session-only in-memory store
sessions: Dict[str, Dict[str, Any]] = {}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ----------------------------
# Serve frontend
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    path = os.path.join(BASE_DIR, "favicon.ico")
    if os.path.exists(path):
        return send_from_directory(".", "favicon.ico")
    return ("", 204)


# ----------------------------
# Helpers
# ----------------------------
def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.5

def rough_token_count(s: str) -> int:
    return max(1, len(s) // 4)

def get_session(user_id: str) -> Dict[str, Any]:
    if user_id not in sessions:
        sessions[user_id] = {"name": "Friend", "history": [], "updated": time.time()}
    sessions[user_id]["updated"] = time.time()
    return sessions[user_id]

def append_history(user_id: str, role: str, content: str) -> None:
    sess = get_session(user_id)
    hist: List[Dict[str, str]] = sess["history"]
    hist.append({"role": role, "content": content})
    if len(hist) > 30:
        del hist[:-30]


# ----------------------------
# Provider adapters
# ----------------------------
def _anthropic_headers() -> Dict[str, str]:
    # Anthropic requires anthropic-version
    if not LLM_API_VERSION:
        raise RuntimeError("Missing required env var for Anthropic: LLM_API_VERSION")
    return {
        "x-api-key": LLM_API_KEY,
        "anthropic-version": LLM_API_VERSION,
        "content-type": "application/json",
    }

def _anthropic_payload(system_prompt: str, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
    # messages must be: [{"role":"user"/"assistant","content":"..."}]
    return {
        "model": LLM_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }

def _openai_compat_headers() -> Dict[str, str]:
    return {
        "authorization": f"Bearer {LLM_API_KEY}",
        "content-type": "application/json",
    }

def _openai_compat_payload(system_prompt: str, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
    # OpenAI-compatible expects system as a message
    full = [{"role": "system", "content": system_prompt}] + messages
    return {
        "model": LLM_MODEL,
        "messages": full,
        "max_tokens": max_tokens,
    }

def _extract_text(provider: str, data: Dict[str, Any]) -> str:
    if provider == "anthropic":
        # content = [{type:"text", text:"..."}]
        out = []
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
        return "".join(out).strip()

    # openai-compatible: choices[0].message.content
    if provider == "openai_compat":
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""

    return ""


# ----------------------------
# Generic LLM call (anonymous)
# ----------------------------
def call_llm(system_prompt, messages, max_tokens=700):

    conversation = system_prompt + "\n\n"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            conversation += f"User: {content}\n"
        else:
            conversation += f"Assistant: {content}\n"

    response = chat_model.generate_content(conversation)
    print("USING KEY:", LLM_API_KEY[:15])
    print("MODEL:", chat_model)

    return response.text


def llm_emotion_read(text: str) -> Dict[str, Any]:
    system = (
        "You are an emotion classifier. "
        "Output ONLY a compact JSON object with keys: "
        'label (one of: "sadness","anxiety","anger","joy","neutral","mixed") '
        "and intensity (0..1). No other text."
    )
    raw = call_llm(system, [{"role": "user", "content": text.strip()}], max_tokens=120)

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"label": "mixed", "intensity": 0.5}

    js = m.group(0)
    label = "mixed"
    intensity = 0.5

    lm = re.search(r'"label"\s*:\s*"([^"]+)"', js)
    if lm:
        cand = lm.group(1).strip().lower()
        if cand in {"sadness", "anxiety", "anger", "joy", "neutral", "mixed"}:
            label = cand

    im = re.search(r'"intensity"\s*:\s*([0-9]*\.?[0-9]+)', js)
    if im:
        intensity = clamp01(im.group(1))

    return {"label": label, "intensity": intensity}


STYLE_GUIDE = {
    "sadness": "Warm Companion",
    "anxiety": "Grounding Guide",
    "anger": "De-escalator + Boundary",
    "joy": "Energizer",
    "neutral": "Clear Helper",
    "mixed": "Mixed / Uncertain Mode",
}

def build_system_prompt(user_name: str, emotion_label: str, intensity: float,
                        uncertainty_mode: bool, clarify_mode: bool) -> str:
    intensity = clamp01(intensity)
    style = STYLE_GUIDE.get(emotion_label, "Clear Helper")

    if intensity >= 0.7:
        intensity_instr = "HIGH intensity: slow down, short sentences, minimal options, grounding tone."
    elif intensity >= 0.3:
        intensity_instr = "MEDIUM intensity: brief validation then 1–2 helpful steps/options."
    else:
        intensity_instr = "LOW intensity: normal helpful tone, keep it concise."

    uncertainty_instr = ""
    if uncertainty_mode or clarify_mode:
        uncertainty_instr = (
            "UNCERTAINTY MODE: do not assert emotion labels. "
            "Use calm de-escalation. Ask ONE high-leverage clarifying question. "
            "Offer two paths: 'Venting' or 'Fix it / Plan'."
        )

    return f"""
You are an emotion-first conversational assistant (session-only).

User name: {user_name}

Strict priority order:
1) Emotional safety
2) User intent
3) Clarity & usefulness
4) Tone mirroring

Emotion steering (internal):
- label: {emotion_label}
- style: {style}
- intensity: {intensity:.2f}
- {intensity_instr}

Tone mirroring:
- Mirror the user’s tone with a cap; do not escalate chaos.
- No sarcasm during distress. No playful tone during sadness/anxiety.
- Emojis: max one, only if the user used emojis first.

Rules:
- Do not say “as an AI”.
- Do not claim to be a therapist/doctor.
- Do not reveal internal mechanisms.
- Keep replies concise unless user asks for depth.
- Do not overwhelm with choices.

{uncertainty_instr}
""".strip()


# ----------------------------
# Load models once
# ----------------------------
print("Loading models...")

for p, name in [
    (TEXT_MODEL_PATH, "text model"),
    (TOKENIZER_PATH, "tokenizer"),
    (VOICE_MODEL_PATH, "voice model"),
    (FACE_MODEL_PATH, "face model"),
]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {name}: {p}")

text_model = tf.keras.models.load_model(TEXT_MODEL_PATH, compile=False)

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tok_data = json.load(f)

text_tokenizer = tokenizer_from_json(json.dumps(tok_data))

try:
    text_maxlen = int(text_model.input_shape[1])
    if text_maxlen <= 0:
        text_maxlen = 120
except Exception:
    text_maxlen = 120

voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH, compile=False)
face_model = tf.keras.models.load_model(FACE_MODEL_PATH, compile=False)

# Auto-detect face image size from model input
try:
    face_img_dim = int(face_model.input_shape[1])
    if face_img_dim > 0:
        FACE_IMAGE_SIZE = face_img_dim
except Exception:
    pass

print("Text model input:", text_model.input_shape, "output:", text_model.output_shape)
print("Voice model input:", voice_model.input_shape, "output:", voice_model.output_shape)
print("Face model input:", face_model.input_shape, "output:", face_model.output_shape)


# ----------------------------
# Emotion model predictors
# ----------------------------
def predict_text_emotion(text: str) -> Dict[str, Any]:
    seq = text_tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=text_maxlen, padding="post", truncating="post")
    probs = text_model.predict(x, verbose=0)[0]
    probs = np.asarray(probs).astype(np.float32)

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = TEXT_LABELS[idx] if idx < len(TEXT_LABELS) else "neutral"
    intensity = clamp01(conf)

    return {
        "label": label,
        "confidence": clamp01(conf),
        "intensity": intensity,
        "probs": probs.tolist()
    }


def ensure_wav(input_path: str) -> str:
    if input_path.lower().endswith(".wav"):
        return input_path

    out_path = input_path + ".wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(TARGET_SR), out_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install it: sudo apt install ffmpeg")
    except subprocess.CalledProcessError:
        raise RuntimeError("ffmpeg failed to convert audio. Unsupported/bad audio?")


def preprocess_voice_to_526x1(audio_path: str) -> np.ndarray:
    wav_path = ensure_wav(audio_path)
    audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=TARGET_SR, n_mfcc=N_MFCC)
    feat = mfcc.T.reshape(-1)

    if feat.size < VOICE_TARGET_LEN:
        feat = np.pad(feat, (0, VOICE_TARGET_LEN - feat.size))
    else:
        feat = feat[:VOICE_TARGET_LEN]

    return feat.astype(np.float32).reshape(1, VOICE_TARGET_LEN, 1)


def predict_voice_emotion(audio_path: str) -> Dict[str, Any]:
    x = preprocess_voice_to_526x1(audio_path)
    probs = voice_model.predict(x, verbose=0)[0]
    probs = np.asarray(probs).astype(np.float32)

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = VOICE_LABELS[idx] if idx < len(VOICE_LABELS) else "neutral"
    intensity = clamp01(conf)

    return {
        "label": label,
        "confidence": clamp01(conf),
        "intensity": intensity,
        "probs": probs.tolist()
    }


def preprocess_face_image(image_path: str) -> np.ndarray:
    print("FACE DEBUG: image_path =", image_path)

    img = cv2.imread(image_path)

    print("FACE DEBUG: img =", img is not None)

    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    print("FACE DEBUG: original shape =", img.shape)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("FACE DEBUG: gray shape =", img_gray.shape)

    img_resized = cv2.resize(img_gray, (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE))

    print("FACE DEBUG: resized shape =", img_resized.shape)

    img_normalized = img_resized.astype(np.float32) / 255.0

    img_input = img_normalized.reshape(
        1,
        FACE_IMAGE_SIZE,
        FACE_IMAGE_SIZE,
        1
    )

    print("FACE DEBUG: final shape =", img_input.shape)

    return img_input.astype(np.float32)


def predict_face_emotion(image_path: str) -> Dict[str, Any]:
    x = preprocess_face_image(image_path)

    probs = face_model.predict(x, verbose=0)[0]
    probs = np.asarray(probs).astype(np.float32)

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = FACE_LABELS[idx] if idx < len(FACE_LABELS) else "Neutral"
    intensity = clamp01(conf)

    print("\n===== FACE MODEL DEBUG =====")
    print("Probabilities:", probs)
    print("Predicted index:", idx)
    print("Predicted label:", label)
    print("Confidence:", conf)
    print("===========================\n")

    return {
        "label": label,
        "confidence": clamp01(conf),
        "intensity": intensity,
        "probs": probs.tolist()
    }


TEXT_TO_STEER = {
    "anger": "anger",
    "fear": "anxiety",
    "joy": "joy",
    "love": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "mixed",
}

VOICE_TO_STEER = {
    "disgust": "anger",
    "angry": "anger",
    "fear": "anxiety",
    "sad": "sadness",
    "happy": "joy",
    "neutral": "neutral",
    "pleasant": "joy",
    "surprise": "mixed",
    "calm": "neutral",
}

FACE_EMOTION_TO_STEER = {
    "Angry": "anger",
    "Disgust": "anger",
    "Fear": "anxiety",
    "Happy": "joy",
    "Sad": "sadness",
    "Surprise": "mixed",
    "Neutral": "neutral",
}


# ----------------------------
# API: register
# ----------------------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "Friend").strip()[:40]

    user_id = uuid.uuid4().hex
    sessions[user_id] = {"name": name, "history": [], "updated": time.time()}
    return jsonify({"user_id": user_id})


# ----------------------------
# API: chat
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_id = (data.get("user_id") or "").strip()
        user_name = (data.get("user_name") or "Friend").strip()[:40]
        message = (data.get("message") or "").strip()

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400
        if not message:
            return jsonify({"error": "Empty message"}), 400

        sess = get_session(user_id)
        sess["name"] = user_name
        history: List[Dict[str, str]] = sess["history"]

        em_raw = predict_text_emotion(message)
        raw_label = em_raw["label"]
        em_label = TEXT_TO_STEER.get(raw_label, "mixed")
        em_conf = clamp01(em_raw["confidence"])
        em_intensity = clamp01(em_raw["intensity"])

        is_long = (len(message) >= LONG_CHARS) or (rough_token_count(message) >= LONG_TOKENS_ROUGH)
        low_conf = em_conf < LOW_CONF
        suspicious_high = em_conf > HIGH_CONF_SUSPICIOUS
        need_verify = low_conf or suspicious_high or is_long

        disagree = False
        clarify_mode = False
        uncertainty_mode = False

        if need_verify:
            llm_em = llm_emotion_read(message)
            llm_label = llm_em.get("label", "mixed")
            llm_intensity = clamp01(llm_em.get("intensity", 0.5))

            disagree = (llm_label != em_label)
            if disagree:
                em_label = llm_label
                em_intensity = llm_intensity
                clarify_mode = True
                uncertainty_mode = True

        system_prompt = build_system_prompt(
            user_name=user_name,
            emotion_label=em_label,
            intensity=em_intensity,
            uncertainty_mode=uncertainty_mode,
            clarify_mode=clarify_mode,
        )

        llm_messages = history + [{"role": "user", "content": message}]
        reply = call_llm(system_prompt, llm_messages, max_tokens=700)

        append_history(user_id, "user", message)
        append_history(user_id, "assistant", reply)

        return jsonify({
            "response": reply,
            "clarify_mode": clarify_mode,
            "buttons": ["Venting", "Fix it / Plan"] if clarify_mode else [],
            "emotion_debug": {
                "raw_label": raw_label,
                "label": em_label,
                "intensity": em_intensity,
                "confidence_model": em_conf,
                "verified": need_verify,
                "disagree": disagree,
            }
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("CHAT ERROR:\n", tb)
        return jsonify({"error": str(e), "trace": tb}), 500


# ----------------------------
# API: voice
# ----------------------------
@app.route("/api/analyze/voice", methods=["POST"])
def analyze_voice():
    tmp_path = None
    try:
        user_id = (request.form.get("user_id") or "").strip()
        user_name = (request.form.get("user_name") or "Friend").strip()[:40]

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400
        if "audio" not in request.files:
            return jsonify({"error": "No audio file field named 'audio'"}), 400

        audio_file = request.files["audio"]

        suffix = ".webm"
        if audio_file.filename and "." in audio_file.filename:
            suffix = "." + audio_file.filename.rsplit(".", 1)[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        sess = get_session(user_id)
        sess["name"] = user_name
        history: List[Dict[str, str]] = sess["history"]

        transcript = "(voice message)"

        em_raw = predict_voice_emotion(tmp_path)
        raw_label = em_raw["label"]
        em_label = VOICE_TO_STEER.get(raw_label, "mixed")
        em_conf = clamp01(em_raw["confidence"])
        em_intensity = clamp01(em_raw["intensity"])

        llm_em = llm_emotion_read(transcript)
        llm_label = llm_em.get("label", "mixed")
        llm_intensity = clamp01(llm_em.get("intensity", 0.5))

        disagree = (llm_label != em_label)
        clarify_mode = False
        uncertainty_mode = False

        if disagree or em_conf < LOW_CONF or em_conf > HIGH_CONF_SUSPICIOUS:
            em_label = llm_label
            em_intensity = llm_intensity
            clarify_mode = True
            uncertainty_mode = True

        system_prompt = build_system_prompt(
            user_name=user_name,
            emotion_label=em_label,
            intensity=em_intensity,
            uncertainty_mode=uncertainty_mode,
            clarify_mode=clarify_mode,
        )

        llm_messages = history + [{"role": "user", "content": transcript}]
        reply = call_llm(system_prompt, llm_messages, max_tokens=700)

        append_history(user_id, "user", transcript)
        append_history(user_id, "assistant", reply)

        return jsonify({
            "transcript": transcript,
            "response": reply,
            "clarify_mode": clarify_mode,
            "buttons": ["Venting", "Fix it / Plan"] if clarify_mode else [],
            "emotion_debug": {
                "raw_label": raw_label,
                "label": em_label,
                "intensity": em_intensity,
                "confidence_model": em_conf,
                "disagree": disagree,
            }
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("VOICE ERROR:\n", tb)
        return jsonify({"error": str(e), "trace": tb}), 500

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            wav_guess = (tmp_path + ".wav") if tmp_path else None
            if wav_guess and os.path.exists(wav_guess):
                os.remove(wav_guess)
        except Exception:
            pass


# ----------------------------
# API: face
# ----------------------------
@app.route("/api/analyze/face", methods=["POST"])
def analyze_face():
    print("FACE API CALLED")
    tmp_path = None
    try:
        user_id = (request.form.get("user_id") or "").strip()
        user_name = (request.form.get("user_name") or "Friend").strip()[:40]

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400
        if "image" not in request.files:
            return jsonify({"error": "No image file field named 'image'"}), 400

        image_file = request.files["image"]

        suffix = ".jpg"
        if image_file.filename and "." in image_file.filename:
            suffix = "." + image_file.filename.rsplit(".", 1)[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            image_file.save(tmp_path)

        sess = get_session(user_id)
        sess["name"] = user_name
        history: List[Dict[str, str]] = sess["history"]

        description = "(facial expression detected)"

        em_raw = predict_face_emotion(tmp_path)
        raw_label = em_raw["label"]
        em_label = FACE_EMOTION_TO_STEER.get(raw_label, "mixed")
        em_conf = clamp01(em_raw["confidence"])
        em_intensity = clamp01(em_raw["intensity"])

        #llm_em = llm_emotion_read(description)
        #llm_label = llm_em.get("label", "mixed")
        #llm_intensity = clamp01(llm_em.get("intensity", 0.5))
        
        llm_label = em_label
        llm_intensity = em_intensity

        disagree = (llm_label != em_label)
        clarify_mode = False
        uncertainty_mode = False

        if disagree or em_conf < LOW_CONF or em_conf > HIGH_CONF_SUSPICIOUS:
            em_label = llm_label
            em_intensity = llm_intensity
            clarify_mode = True
            uncertainty_mode = True

        system_prompt = build_system_prompt(
            user_name=user_name,
            emotion_label=em_label,
            intensity=em_intensity,
            uncertainty_mode=uncertainty_mode,
            clarify_mode=clarify_mode,
        )

        llm_messages = history + [{"role": "user", "content": description}]
        reply = call_llm(system_prompt, llm_messages, max_tokens=700)

        append_history(user_id, "user", description)
        append_history(user_id, "assistant", reply)

        return jsonify({
            "description": description,
            "response": reply,
            "clarify_mode": clarify_mode,
            "buttons": ["Venting", "Fix it / Plan"] if clarify_mode else [],
            "emotion_debug": {
                "raw_label": raw_label,
                "label": em_label,
                "intensity": em_intensity,
                "confidence_model": em_conf,
                "disagree": disagree,
            }
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("FACE ERROR:\n", tb)
        return jsonify({"error": str(e), "trace": tb}), 500

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "sessions": len(sessions)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
