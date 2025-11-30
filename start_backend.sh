#!/bin/bash

# QuadRAG Backend Startup Script

cd "$(dirname "$0")/backend"

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

source .venv/bin/activate

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cat > .env << EOF
# API Keys - Replace with your actual keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
EOF
    echo "📝 Please edit backend/.env and add your API keys, then run this script again."
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Check if API keys are set
if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "your_groq_api_key_here" ]; then
    echo "❌ GROQ_API_KEY not set in .env file"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ OPENAI_API_KEY not set in .env file"
    exit 1
fi

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
    echo "❌ GOOGLE_API_KEY not set in .env file"
    exit 1
fi

echo "✅ Environment variables loaded"
echo "🚀 Starting QuadRAG Backend API..."
echo "📍 Backend will be available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the API server
python api.py

