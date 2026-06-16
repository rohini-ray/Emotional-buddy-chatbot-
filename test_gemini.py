import google.generativeai as genai

API_KEY = "AQ.Ab8RN6JtANw-bucw6xunyLHVWy4TpMHWaLuuylsmuzkM0av1hg"

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

try:
    response = model.generate_content("Hello")
    print(response.text)

except Exception as e:
    print("ERROR:", e)