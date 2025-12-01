#!/bin/bash

# Railway startup script for QuadRAG
set -x  # Enable command tracing

# Set up environment variables with fallbacks
LIBDIR=$(cat /app/libstdcpp_dir.txt 2>/dev/null || echo '/nix/var/nix/profiles/default/lib')
TORCH_LIB=$(cat /app/torch_dir.txt 2>/dev/null || echo '')/lib
export LD_LIBRARY_PATH=$LIBDIR:$TORCH_LIB:/nix/var/nix/profiles/default/lib:/usr/lib:$LD_LIBRARY_PATH

echo "=== QuadRAG Startup Debug ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PORT=${PORT:-8080}"
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

echo "=== Launching FastAPI Application ==="
# Don't use exec so we can see if the startup fails
python api.py 2>&1
