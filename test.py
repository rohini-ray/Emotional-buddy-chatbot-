# test.py

from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load .env file
load_dotenv()

api_key = os.getenv("LLM_API_KEY")

print("Loaded Key:", api_key)

if not api_key:
    print("ERROR: LLM_API_KEY not found in .env")
    exit()

try:
    # Configure Gemini
    genai.configure(api_key=api_key)

    # Create model
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Test request
    response = model.generate_content("Say hello in one sentence.")

    print("\nSUCCESS!")
    print("Response:")
    print(response.text)

except Exception as e:
    print("\nERROR:")
    print(type(e).__name__)
    print(e)