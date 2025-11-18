# 🚀 Quick Start Guide - Run QuadRAG Now!

## Step 1: Install Backend Dependencies

Open Terminal and run:

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/backend

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (this may take 5-10 minutes)
pip install groq openai google-generativeai fastapi uvicorn[standard] pydantic pydantic-settings python-dotenv python-multipart loguru moviepy pillow sentence-transformers transformers torch numpy aiofiles pixeltable
```

**OR** use the setup script:

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/backend
./setup_and_test.sh
```

## Step 2: Verify API Keys

Your API keys are already configured in `backend/.env`. Verify they're loaded:

```bash
cd backend
source .venv/bin/activate
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('GROQ:', os.getenv('GROQ_API_KEY')[:10] + '...'); print('OpenAI:', os.getenv('OPENAI_API_KEY')[:10] + '...'); print('Google:', os.getenv('GOOGLE_API_KEY')[:10] + '...')"
```

## Step 3: Start Backend Server

In Terminal 1:

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/backend
source .venv/bin/activate
python api.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 4: Test Backend (Optional)

In a new terminal:

```bash
curl http://localhost:8000/health
```

Should return: `{"status":"ok","message":"QuadRAG API is healthy"}`

## Step 5: Install Frontend Dependencies

In Terminal 2:

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/frontend
pip install streamlit requests Pillow python-dotenv
```

## Step 6: Start Streamlit UI

In Terminal 2 (same terminal as Step 5):

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`

## Step 7: Use QuadRAG! 🎬

1. **Set Domain Context**
   - Enter: "Capture emotions and facial expressions"
   - Click "Set Context"

2. **Upload a Video**
   - Click "📤 Upload New Video"
   - Choose a video file (MP4, AVI, MOV)
   - Click "Upload and Process"
   - Wait 2-3 minutes for processing

3. **Ask Questions**
   - Type: "What happens in the video?"
   - Or: "What emotions are shown?"
   - View citations to see timestamps!

## Troubleshooting

### If backend won't start:
```bash
# Check Python path
cd backend
source .venv/bin/activate
python --version  # Should be 3.10+

# Check if dependencies installed
pip list | grep groq
pip list | grep fastapi
```

### If frontend can't connect:
- Make sure backend is running on port 8000
- Check: `curl http://localhost:8000/health`

### If video processing fails:
- Make sure FFmpeg is installed: `ffmpeg -version`
- Try a shorter video first (< 2 minutes)

## Quick Commands Reference

```bash
# Terminal 1 - Backend
cd backend && source .venv/bin/activate && python api.py

# Terminal 2 - Frontend  
cd frontend && streamlit run app.py

# Test API
curl http://localhost:8000/health
```

## Need Help?

- See `GETTING_STARTED.md` for detailed instructions
- See `ARCHITECTURE.md` for system details
- Check backend terminal for error messages

Happy video understanding! 🎬✨


