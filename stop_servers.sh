#!/bin/bash

# Stop QuadRAG Servers

echo "🛑 Stopping QuadRAG servers..."

# Stop backend
BACKEND_PID=$(lsof -ti:8000)
if [ ! -z "$BACKEND_PID" ]; then
    echo "Stopping backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID
    sleep 2
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo "Force killing backend..."
        kill -9 $BACKEND_PID
    fi
    echo "✅ Backend stopped"
else
    echo "ℹ️  Backend not running"
fi

# Stop frontend
FRONTEND_PID=$(lsof -ti:8501)
if [ ! -z "$FRONTEND_PID" ]; then
    echo "Stopping frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID
    sleep 2
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo "Force killing frontend..."
        kill -9 $FRONTEND_PID
    fi
    echo "✅ Frontend stopped"
else
    echo "ℹ️  Frontend not running"
fi

# Also kill any remaining python/streamlit processes
pkill -f "python.*api.py" 2>/dev/null
pkill -f "streamlit run" 2>/dev/null

echo ""
echo "✅ All servers stopped"

