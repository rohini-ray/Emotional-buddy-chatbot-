import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_response(
        message,
        emotion):

    prompt = f"""
User Emotion: {emotion}

User Message:
{message}

Respond in a supportive,
empathetic and helpful way.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content