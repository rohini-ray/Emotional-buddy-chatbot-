import google.generativeai as genai

genai.configure(api_key="AQ.Ab8RN6JtANw-bucw6xunyLHVWy4TpMHWaLuuylsmuzkM0av1hg")

for m in genai.list_models():
    print(m.name)