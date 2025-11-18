#!/bin/bash

# QuadRAG Setup and Test Script

echo "🚀 Setting up QuadRAG..."

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install groq openai google-generativeai fastapi uvicorn[standard] pydantic pydantic-settings python-dotenv python-multipart loguru moviepy pillow sentence-transformers transformers torch numpy aiofiles pixeltable

echo "✅ Dependencies installed!"

# Test API keys
echo ""
echo "🔑 Testing API keys..."
python3 << 'EOF'
import os
from dotenv import load_dotenv

load_dotenv()

keys = {
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
}

for key_name, key_value in keys.items():
    if key_value:
        print(f"✅ {key_name}: {'*' * 20}...{key_value[-4:]}")
    else:
        print(f"❌ {key_name}: NOT SET")

print("\n✅ API keys loaded!")
EOF

echo ""
echo "🎬 Ready to start!"
echo ""
echo "To start the backend:"
echo "  python api.py"
echo ""
echo "To test the API:"
echo "  curl http://localhost:8000/health"


