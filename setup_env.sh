#!/bin/bash

# QuadRAG Environment Setup Script

echo "🚀 Setting up QuadRAG environment..."

# Create data directories
mkdir -p data/videos data/cache

# Setup backend
echo "📦 Setting up backend..."
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
cd ..

# Setup frontend
echo "🎨 Setting up frontend..."
cd frontend
pip install -r requirements.txt
cd ..

# Create .env from example if it doesn't exist
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "⚠️  Please edit backend/.env and add your API keys"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit backend/.env and add your API keys"
echo "2. Start backend: cd backend && uvicorn api:app --reload"
echo "3. Start frontend: cd frontend && streamlit run app.py"


