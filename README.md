# QuadRAG: A Four-Index Multimodal Retrieval-Augmented Framework for Video Understanding

QuadRAG is a training-free video comprehension system that uses four distinct semantic indexes to enable rich, context-aware question answering about video content.

## Architecture

QuadRAG creates and leverages four parallel semantic indexes:

1. **Image Index** - Raw video frames with CLIP embeddings
2. **Audio Index** - Transcribed spoken dialogue with text embeddings
3. **Description Index** - Frame descriptions with text embeddings
4. **Domain Captions Index** - Context-specific captions based on user-provided domain

## Features

- 🎬 Multi-modal video processing (visual, audio, text)
- 🔍 Four-way retrieval with intelligent fusion
- 🎯 Dynamic domain-specific caption generation
- 💬 Conversational interface with Streamlit UI
- ⚡ Powered by Groq LLMs and Pixeltable vector database

## Installation

### Backend Setup

```bash
cd backend
pip install -e .
```

### Frontend Setup

```bash
cd frontend
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and add your API keys:

```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Usage

### Start Backend API

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Start Streamlit UI

```bash
cd frontend
streamlit run app.py
```

### Using QuadRAG

1. Open the Streamlit interface
2. Set your domain context (e.g., "Capture emotions and facial expressions")
3. Upload a video
4. Wait for processing to complete
5. Ask questions about the video

## Project Structure

```
QuadRag/
├── backend/
│   ├── src/quadrag/         # Core QuadRAG modules
│   ├── api.py              # FastAPI backend
│   └── pyproject.toml
├── frontend/
│   ├── app.py              # Streamlit UI
│   └── requirements.txt
├── data/
│   ├── videos/             # Uploaded videos
│   └── cache/              # Pixeltable cache
└── README.md
```

## Technology Stack

- **Vector Database**: Pixeltable
- **LLM**: Groq (Llama 4 Scout/Maverick)
- **Vision Models**: GPT-4o-mini
- **Embeddings**: CLIP (images), Gemini (text)
- **Backend**: FastAPI
- **Frontend**: Streamlit

## License

MIT License


