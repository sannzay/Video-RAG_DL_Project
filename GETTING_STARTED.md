# Getting Started with QuadRAG

This guide will help you set up and run QuadRAG on your local machine.

## Prerequisites

- Python 3.10 or higher
- FFmpeg (for video processing)
- Git

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Installation

### 1. Clone the Repository

```bash
cd /path/to/QuadRag
```

### 2. Set Up Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Configure API Keys

Create a `.env` file in the `backend` directory:

```bash
cd backend
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

**Getting API Keys:**

- **Groq**: Sign up at [console.groq.com](https://console.groq.com)
- **OpenAI**: Get key from [platform.openai.com](https://platform.openai.com)
- **Google**: Create API key at [makersuite.google.com](https://makersuite.google.com)

### 4. Set Up Frontend

```bash
cd ../frontend

# Install dependencies
pip install -r requirements.txt
```

## Running QuadRAG

### Start Backend Server

In a terminal, from the `backend` directory:

```bash
cd backend
source .venv/bin/activate  # Activate virtual environment
python api.py
```

The backend will start on `http://localhost:8000`

You can verify it's running by visiting: `http://localhost:8000/health`

### Start Frontend UI

In a **new terminal**, from the `frontend` directory:

```bash
cd frontend
streamlit run app.py
```

The Streamlit UI will open automatically in your browser at `http://localhost:8501`

## Using QuadRAG

### 1. Set Domain Context

When you first open the app, you'll be prompted to set a domain context. Examples:

- "Capture emotions and facial expressions"
- "Identify objects and their locations"
- "Focus on text and written content"
- "Analyze body language and gestures"

### 2. Upload a Video

- Click "📤 Upload New Video"
- Choose a video file (MP4, AVI, MOV, MKV)
- Click "Upload and Process"
- Wait for processing to complete (this creates the 4 indexes)

### 3. Chat with Your Video

Once processing is complete:

- Type your question in the chat input
- The system will search all 4 indexes
- You'll get an answer with citations showing timestamps

### 4. View Citations

Click "📎 View Citations" to see:
- Which index the information came from
- Timestamp in the video
- The actual content retrieved

## Troubleshooting

### Backend won't start

**Error: "No module named 'quadrag'"**
- Make sure you're in the backend directory
- Activate the virtual environment
- Run `pip install -e .`

**Error: "API key not found"**
- Check that `.env` file exists in `backend/` directory
- Verify all API keys are set correctly
- No quotes needed around the values

### Video processing fails

**Error: "ffmpeg not found"**
- Install ffmpeg (see prerequisites)
- Verify: `ffmpeg -version`

**Error: "Failed to process video"**
- Check video format is supported
- Try re-encoding: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`
- Check backend logs for detailed error

### Streamlit connection error

**Error: "Connection refused"**
- Make sure backend is running first
- Check backend is on `http://localhost:8000`
- Try restarting both backend and frontend

### Out of memory

If processing large videos:
- Reduce `SPLIT_FRAMES_COUNT` in `.env` (default: 45)
- Reduce `AUDIO_CHUNK_LENGTH` (default: 10)
- Process shorter video clips

## Project Structure

```
QuadRag/
├── backend/
│   ├── src/quadrag/          # Core modules
│   │   ├── video/            # Video processing & indexing
│   │   ├── retrieval/        # Search & fusion
│   │   └── generation/       # RAG generation
│   ├── api.py                # FastAPI server
│   └── .env                  # Configuration
├── frontend/
│   └── app.py                # Streamlit UI
└── data/
    ├── videos/               # Uploaded videos
    └── cache/                # Pixeltable cache
```

## Next Steps

- Try different domain contexts
- Upload multiple videos
- Compare how different indexes contribute
- Check the citations to understand retrieval

## Support

For issues or questions:
- Check the logs in backend terminal
- Review the API documentation at `http://localhost:8000/docs`
- Refer to the main README.md

Happy video understanding! 🎬


