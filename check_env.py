import os
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Dotenv loaded.")
except ImportError:
    print("Dotenv not found.")

google = os.getenv("GOOGLE_API_KEY")
openai = os.getenv("OPENAI_API_KEY")

if google:
    print(f"GOOGLE_API_KEY found: {google[:5]}...")
else:
    print("GOOGLE_API_KEY NOT found.")

if openai:
    print(f"OPENAI_API_KEY found: {openai[:5]}...")
else:
    print("OPENAI_API_KEY NOT found.")
