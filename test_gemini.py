import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file in backend directory
backend_env_path = os.path.join(os.path.dirname(__file__), "network-attack-backend", ".env")
load_dotenv(backend_env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    exit(1)

genai.configure(api_key=api_key)
print("genai version:", getattr(genai, "__version__", "unknown"))

model = genai.GenerativeModel("gemini-2.0-flash")
resp = model.generate_content("Say hello, this is a test from Priyanshu.")
print(resp.text)
