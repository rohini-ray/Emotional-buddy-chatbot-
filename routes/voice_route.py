import os
import tempfile

from flask import Blueprint
from flask import request
from flask import jsonify

from services.voice_emotion import (
    predict_voice_emotion
)

voice_bp = Blueprint(
    "voice_bp",
    __name__
)


@voice_bp.route(
    "/api/analyze/voice",
    methods=["POST"]
)
def analyze_voice():

    temp_path = None

    try:

        if "audio" not in request.files:
            return jsonify({
                "error": "No audio uploaded"
            }), 400

        audio_file = request.files["audio"]

        suffix = ".wav"

        if "." in audio_file.filename:
            suffix = "." + audio_file.filename.split(".")[-1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp_path = temp.name
            audio_file.save(temp_path)

        result = predict_voice_emotion(
            temp_path
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)