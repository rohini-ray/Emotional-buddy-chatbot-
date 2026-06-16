import os
import tempfile

from flask import Blueprint
from flask import request
from flask import jsonify

from services.text_emotion import (
    predict_text_emotion
)

from services.voice_emotion import (
    predict_voice_emotion
)

from services.facial_emotion import (
    predict_face_emotion
)

from services.fusion import (
    fuse_emotions
)

multimodal_bp = Blueprint(
    "multimodal_bp",
    __name__
)


@multimodal_bp.route(
    "/api/fusion",
    methods=["POST"]
)
def fusion():

    image_path = None
    audio_path = None

    try:

        text = request.form.get(
            "text",
            ""
        )

        text_result = None
        voice_result = None
        face_result = None

        # TEXT
        if text:
            text_result = predict_text_emotion(
                text
            )

        # AUDIO
        if "audio" in request.files:

            audio_file = request.files["audio"]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as temp:

                audio_path = temp.name
                audio_file.save(audio_path)

            voice_result = predict_voice_emotion(
                audio_path
            )

        # IMAGE
        if "image" in request.files:

            image_file = request.files["image"]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp:

                image_path = temp.name
                image_file.save(image_path)

            face_result = predict_face_emotion(
                image_path
            )

        final_emotion = fuse_emotions(
            text_result=text_result,
            voice_result=voice_result,
            face_result=face_result
        )

        return jsonify({
            "text": text_result,
            "voice": voice_result,
            "face": face_result,
            "fusion": final_emotion
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if image_path and os.path.exists(
            image_path
        ):
            os.remove(image_path)

        if audio_path and os.path.exists(
            audio_path
        ):
            os.remove(audio_path)