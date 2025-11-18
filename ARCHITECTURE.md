# QuadRAG Architecture

## Overview

QuadRAG is a training-free multimodal video understanding system that uses four parallel semantic indexes to enable rich, context-aware question answering about video content.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Domain       │  │ Video        │  │ Chat         │      │
│  │ Context      │  │ Upload       │  │ Interface    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Video Processing Pipeline               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│   │
│  │  │ Extract │→ │ Process │→ │ Create  │→ │  Store  ││   │
│  │  │ Frames  │  │ Audio   │  │ Indexes │  │  Index  ││   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘│   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Four Semantic Indexes                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│   │
│  │  │  Image  │  │  Audio  │  │  Desc   │  │ Domain  ││   │
│  │  │  Index  │  │  Index  │  │  Index  │  │ Index   ││   │
│  │  │  CLIP   │  │ Gemini  │  │ Gemini  │  │ Gemini  ││   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘│   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              RAG Pipeline                            │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│   │
│  │  │ Search  │→ │  Fuse   │→ │Generate │→ │ Return  ││   │
│  │  │ Indexes │  │ Results │  │  with   │  │ Answer  ││   │
│  │  │         │  │         │  │  Groq   │  │         ││   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘│   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pixeltable Storage                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Video Tables │ Frame Views │ Audio Views │ Indexes  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Video Processing Pipeline

**Location**: `backend/src/quadrag/video/`

- **processor.py**: Main video processor
  - Manages video ingestion
  - Creates Pixeltable tables
  - Coordinates index creation

- **indexer.py**: Creates the four semantic indexes
  - Image Index: CLIP embeddings of raw frames
  - Audio Index: Transcribed speech with text embeddings
  - Description Index: Generic frame descriptions
  - Domain Index: Context-specific captions

### 2. Four Semantic Indexes

#### Image Index
- **Purpose**: Visual similarity search
- **Process**: 
  1. Extract frames (default: 45 frames)
  2. Resize to 1024x768
  3. Generate CLIP embeddings
- **Model**: `openai/clip-vit-base-patch32`
- **Use Case**: Find visually similar moments

#### Audio Index
- **Purpose**: Search spoken content
- **Process**:
  1. Extract audio stream
  2. Split into 10-second chunks
  3. Transcribe with Whisper
  4. Generate text embeddings
- **Models**: 
  - Transcription: `whisper-1`
  - Embeddings: `text-embedding-3-small`
- **Use Case**: Find what was said

#### Description Index
- **Purpose**: Semantic understanding of scenes
- **Process**:
  1. Use same frames as Image Index
  2. Generate descriptions with GPT-4o-mini
  3. Embed descriptions
- **Models**:
  - Description: `gpt-4o-mini`
  - Embeddings: `text-embedding-3-small`
- **Use Case**: Understand what's happening

#### Domain Index
- **Purpose**: Context-specific analysis
- **Process**:
  1. User provides domain context
  2. Generate focused captions
  3. Embed domain-specific captions
- **Models**:
  - Captions: `gpt-4o-mini`
  - Embeddings: `text-embedding-3-small`
- **Use Case**: Domain-specific queries

### 3. Retrieval System

**Location**: `backend/src/quadrag/retrieval/`

- **search_engine.py**: Multi-index search
  - Searches each index independently
  - Returns top-k results per index
  - Supports text and image queries

- **fusion.py**: Result fusion
  - Normalizes similarity scores
  - Applies index-specific weights:
    - Audio: 0.3
    - Image: 0.2
    - Description: 0.25
    - Domain: 0.25
  - Deduplicates by timestamp
  - Returns top-k fused results

### 4. RAG Generation

**Location**: `backend/src/quadrag/generation/`

- **rag_generator.py**: Answer generation
  - Builds context from retrieved results
  - Formats prompt with all modalities
  - Calls Groq LLM (Llama 3.3 70B)
  - Returns answer with citations

### 5. API Layer

**Location**: `backend/api.py`

**Endpoints**:
- `POST /upload-video`: Upload video file
- `POST /process-video`: Start processing
- `GET /video/{id}/status`: Check status
- `POST /set-domain-context`: Create domain index
- `POST /chat`: Query with RAG
- `GET /videos`: List all videos

### 6. Frontend UI

**Location**: `frontend/app.py`

**Features**:
- Domain context dialog
- Video upload with progress
- Video library sidebar
- Chat interface
- Citation display with timestamps

## Data Flow

### Video Processing Flow

```
1. User uploads video
   ↓
2. FastAPI saves video file
   ↓
3. VideoProcessor creates Pixeltable table
   ↓
4. VideoIndexer creates 3 base indexes (parallel)
   ├─ Image Index (CLIP embeddings)
   ├─ Audio Index (transcribe + embed)
   └─ Description Index (describe + embed)
   ↓
5. User sets domain context
   ↓
6. VideoIndexer creates Domain Index
```

### Query Flow

```
1. User asks question
   ↓
2. VideoSearchEngine searches all 4 indexes
   ├─ Image Index
   ├─ Audio Index
   ├─ Description Index
   └─ Domain Index
   ↓
3. ResultFusion combines results
   ├─ Normalize scores
   ├─ Apply weights
   ├─ Deduplicate
   └─ Rank top-k
   ↓
4. RAGGenerator builds context
   ├─ Format retrieved content
   ├─ Add domain context
   └─ Create prompt
   ↓
5. Groq LLM generates answer
   ↓
6. Return answer + citations
```

## Key Design Decisions

### Why Four Indexes?

1. **Image Index**: Captures visual content that may not be described in audio
2. **Audio Index**: Captures spoken information not visible in frames
3. **Description Index**: Provides general understanding of scenes
4. **Domain Index**: Enables specialized, context-aware analysis

### Why Pixeltable?

- Built for multimodal data
- Efficient incremental processing
- Native embedding index support
- Easy similarity search

### Why Groq?

- Fast inference (Llama 3.3 70B)
- Good reasoning capabilities
- Cost-effective
- Generous free tier

### Dynamic Domain Captions

- Generated per-session, not pre-computed
- Allows flexible domain adaptation
- Users can change context for same video
- Enables specialized use cases

## Performance Considerations

### Video Processing Time

For a 5-minute video:
- Frame extraction: ~10s
- Audio extraction + transcription: ~30s
- Image descriptions: ~60s (45 frames × ~1.3s)
- CLIP embeddings: ~5s
- **Total: ~2 minutes**

Domain index adds ~60s when created.

### Query Time

- Multi-index search: ~100ms
- Result fusion: ~10ms
- LLM generation: ~2-5s
- **Total: ~3-5s per query**

### Optimization Strategies

1. **Reduce frames**: Lower `SPLIT_FRAMES_COUNT` (trade-off: coverage)
2. **Parallel processing**: Indexes created concurrently
3. **Caching**: Pixeltable caches computed columns
4. **Batch processing**: Multiple queries reuse same indexes

## Future Enhancements

1. **OCR Index**: Extract and search text in video
2. **Object Detection Index**: Track specific objects
3. **Scene Segmentation**: Group related frames
4. **Video Summarization**: Generate video summaries
5. **Multi-video Search**: Search across video library
6. **Streaming**: Real-time processing for live videos

## Security Considerations

1. **API Keys**: Stored in `.env`, never committed
2. **File Upload**: Validate file types and sizes
3. **CORS**: Configure for production
4. **Rate Limiting**: Add for production deployment
5. **Authentication**: Add user authentication

## Deployment

### Development
- Backend: `python api.py`
- Frontend: `streamlit run app.py`

### Production
- Backend: Gunicorn/Uvicorn behind Nginx
- Frontend: Streamlit Cloud or Docker
- Storage: Cloud storage for videos
- Database: PostgreSQL for metadata
- Cache: Redis for session management


