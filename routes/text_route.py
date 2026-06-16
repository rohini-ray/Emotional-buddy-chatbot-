from flask import Blueprint, request, jsonify

from services.text_emotion import predict_text_emotion
from services.groq_service import generate_response

text_bp = Blueprint(
    "text_bp",
    __name__
)


@text_bp.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    try:
        data = request.get_json()

        message = data.get(
            "message",
            ""
        )

        if not message:
            return jsonify({
                "error": "Message required"
            }), 400

        text_result = predict_text_emotion(
            message
        )

        emotion = text_result["emotion"]

        reply = generate_response(
            message,
            emotion
        )

        return jsonify({
            "message": message,
            "emotion": emotion,
            "confidence": text_result["confidence"],
            "response": reply
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500