VOICE_MAP = {
    "angry": "anger",
    "disgust": "anger",
    "fear": "fear",
    "sad": "sadness",
    "happy": "joy",
    "pleasant": "joy",
    "neutral": "neutral",
    "calm": "neutral",
    "surprise": "surprise"
}

TEXT_MAP = {
    "anger": "anger",
    "fear": "fear",
    "joy": "joy",
    "love": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise"
}

FACE_MAP = {
    "angry": "anger",
    "disgust": "anger",
    "fear": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral"
}


def fuse_emotions(
        text_result=None,
        voice_result=None,
        face_result=None):

    scores = {}

    if text_result:
        emotion = TEXT_MAP[text_result["emotion"]]
        scores[emotion] = scores.get(
            emotion,
            0
        ) + text_result["confidence"]

    if voice_result:
        emotion = VOICE_MAP[voice_result["emotion"]]
        scores[emotion] = scores.get(
            emotion,
            0
        ) + voice_result["confidence"]

    if face_result:
        emotion = FACE_MAP[face_result["emotion"]]
        scores[emotion] = scores.get(
            emotion,
            0
        ) + face_result["confidence"]

    if not scores:
        return {
            "emotion": "neutral",
            "confidence": 0
        }

    final_emotion = max(
        scores,
        key=scores.get
    )

    return {
        "emotion": final_emotion,
        "confidence": round(
            scores[final_emotion] /
            sum(scores.values()),
            3
        )
    }