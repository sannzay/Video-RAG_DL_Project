# QuadRAG Quick Start Guide

Get QuadRAG up and running in 5 minutes!

## Prerequisites

```bash
# Check Python version (need 3.10+)
python --version

# Check FFmpeg (needed for video processing)
ffmpeg -version

# If FFmpeg not installed:
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
```

## Step 1: Install Backend (2 minutes)

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/backend

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# OR: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Step 2: Configure API Keys (1 minute)

Create `backend/.env` file:

```bash
cd backend
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
EOF
```

**Get API Keys:**
- Groq: https://console.groq.com (free tier: 500K tokens/day)
- OpenAI: https://platform.openai.com ($5 free on signup)
- Google: https://makersuite.google.com (optional)

Edit the `.env` file with your actual keys.

## Step 3: Install Frontend (1 minute)

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/frontend

pip install -r requirements.txt
```

## Step 4: Start Backend (30 seconds)

Open Terminal 1:

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/backend
source .venv/bin/activate
python api.py
```

You should see: `Uvicorn running on http://0.0.0.0:8000`

## Step 5: Start Frontend (30 seconds)

Open Terminal 2:

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/frontend
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`

## Step 6: Use QuadRAG! 🎬

1. **Set Domain Context**
   - Enter: "Capture emotions and facial expressions"
   - Click "Set Context"

2. **Upload Video**
   - Click "📤 Upload New Video"
   - Choose a video file (MP4, AVI, MOV)
   - Click "Upload and Process"
   - Wait ~2-3 minutes for processing

3. **Ask Questions**
   - Type: "What emotions does the person show?"
   - Or: "What happens at the beginning?"
   - Or: "Describe the main events"

4. **View Citations**
   - Click "📎 View Citations" to see timestamps
   - Each citation shows: source index + timestamp

## Example Queries

### Emotional Analysis
- "What emotions are displayed?"
- "When does the person look happy?"
- "Describe the emotional tone"

### Content Questions
- "What is the main topic discussed?"
- "Summarize what happens"
- "What objects are visible?"

### Temporal Questions
- "What happens in the first minute?"
- "When does the music start?"
- "What changes throughout the video?"

## Troubleshooting

### Backend won't start
```bash
# Check you're in the right directory
pwd  # Should end in /backend

# Activate virtual environment
source .venv/bin/activate

# Check API keys
cat .env  # Should show your keys
```

### "Module not found" error
```bash
cd backend
pip install -e .
```

### Video processing fails
```bash
# Check FFmpeg
ffmpeg -version

# Try a different video format
# Or re-encode: ffmpeg -i input.mp4 -c:v libx264 output.mp4
```

### Frontend can't connect
```bash
# Make sure backend is running on port 8000
curl http://localhost:8000/health
# Should return: {"status":"ok","message":"QuadRAG API is healthy"}
```

## Next Steps

- Try different domain contexts
- Upload multiple videos
- Check out `GETTING_STARTED.md` for advanced usage
- Read `ARCHITECTURE.md` to understand the system

## Quick Commands Reference

```bash
# Start backend
cd backend && source .venv/bin/activate && python api.py

# Start frontend (in new terminal)
cd frontend && streamlit run app.py

# Test API
curl http://localhost:8000/health

# Check logs
# Backend logs appear in Terminal 1
# Frontend logs appear in Terminal 2
```

## Performance Tips

- **Faster processing**: Reduce `SPLIT_FRAMES_COUNT` in `.env` (default: 45)
- **Less memory**: Process shorter videos (< 5 minutes)
- **Better results**: Use high-quality videos

## Support

- **Full guide**: See `GETTING_STARTED.md`
- **Architecture**: See `ARCHITECTURE.md`
- **API docs**: Visit `http://localhost:8000/docs`

Happy video understanding! 🎬✨


