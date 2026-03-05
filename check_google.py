import os
import sys

try:
    import google.generativeai as genai
    print("Package 'google.generativeai' found.")
except ImportError:
    print("Package 'google.generativeai' NOT found.")
    sys.exit(1)

key = os.getenv("GOOGLE_API_KEY")
if not key:
    print("Key not found in env.")
    sys.exit(1)

genai.configure(api_key=key)
try:
    print("Listing models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
    
    # model = genai.GenerativeModel('gemini-pro')
    # response = model.generate_content("Hello")
    # print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Error calling API: {e}")
