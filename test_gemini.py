import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

print(f"Key found: {key[:5]}...")

genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-1.5-flash')

try:
    print("Sending prompt...")
    response = model.generate_content("Hello, are you working?")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
