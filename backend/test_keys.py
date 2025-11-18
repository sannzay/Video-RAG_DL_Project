"""Quick test to verify API keys are loaded correctly."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

print("🔑 Testing API Keys...")
print("=" * 50)

keys = {
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
}

all_ok = True
for key_name, key_value in keys.items():
    if key_value:
        masked = key_value[:10] + "..." + key_value[-4:] if len(key_value) > 14 else "***"
        print(f"✅ {key_name}: {masked}")
    else:
        print(f"❌ {key_name}: NOT SET")
        all_ok = False

print("=" * 50)
if all_ok:
    print("✅ All API keys are configured!")
    print("\nYou can now start the backend with:")
    print("  python api.py")
else:
    print("❌ Some API keys are missing!")
    print("Please check backend/.env file")


