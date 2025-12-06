#!/bin/bash

# Railway startup script for QuadRAG
set -x  # Enable command tracing

# Set up environment variables with fallbacks
LIBDIR=$(cat /app/libstdcpp_dir.txt 2>/dev/null || echo '/nix/var/nix/profiles/default/lib')
TORCH_LIB=$(cat /app/torch_dir.txt 2>/dev/null || echo '')/lib
export LD_LIBRARY_PATH=$LIBDIR:$TORCH_LIB:/nix/var/nix/profiles/default/lib:/usr/lib:$LD_LIBRARY_PATH

echo "=== QuadRAG Startup Debug ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PORT environment variable: ${PORT:-NOT_SET}"
echo "Defaulting PORT to: ${PORT:-8080}"
echo "DATABASE_URL: ${DATABASE_URL:-NOT_SET}"
echo "PWD=$(pwd)"

echo "=== Directory Contents ==="
ls -la /app/
echo "=== Backend Directory ==="
ls -la /app/backend/ 2>/dev/null || echo 'No backend directory found!'

echo "=== Activating Virtual Environment ==="
source /app/venv/bin/activate 2>/dev/null && echo 'Virtualenv activated successfully' || (echo 'Virtualenv activation failed!' && exit 1)

echo "=== Changing to Backend Directory ==="
cd backend 2>/dev/null && echo 'Changed to backend directory' || (echo 'Failed to cd to backend!' && exit 1)

echo "=== Testing Basic Imports ==="
python -c 'import sys; print("Python path:", sys.path[:3]); import numpy as np; print(f"NumPy OK: {np.__version__}"); import torch; print(f"Torch OK: {torch.__version__}")' 2>&1 || (echo 'Basic imports failed!' && exit 1)

echo "=== Testing API Import ==="
python -c 'import sys; sys.path.insert(0, "."); print("Importing api.py..."); import api; print("api.py imported successfully")' 2>&1 || (echo 'API import failed!' && exit 1)

echo "=== Testing FastAPI App Creation ==="
python -c 'import sys; sys.path.insert(0, "."); import api; print(f"FastAPI app created: {api.app.title}")' 2>&1 || (echo 'FastAPI app creation failed!' && exit 1)

echo "=== Testing Pixeltable Import (without database) ==="
python -c 'import pixeltable as pxt; print(f"Pixeltable imported: {pxt.__version__}")' 2>&1 || (echo 'Pixeltable import failed - this might be expected' && true)

echo "=== Checking Pixeltable Home ==="
python -c "import os; home = os.environ.get('PIXELTABLE_HOME', 'NOT_SET'); print(f'PIXELTABLE_HOME: {home}')" 2>&1

echo "=== Launching FastAPI Application ==="
# Don't use exec so we can see if the startup fails
if python api.py 2>&1; then
    echo "FastAPI application exited successfully"
else
    echo "FastAPI application failed with exit code $?"
    exit 1
fi
