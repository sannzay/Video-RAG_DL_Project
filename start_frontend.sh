#!/bin/bash

# QuadRAG Frontend Startup Script

cd "$(dirname "$0")/frontend"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "📦 Installing dependencies..."
    pip install -q -r requirements.txt
    touch .deps_installed
fi

echo "✅ Frontend environment ready"
echo "🚀 Starting QuadRAG Frontend (Streamlit)..."
echo "📍 Frontend will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run app.py --server.port 8501

