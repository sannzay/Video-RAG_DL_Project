# Railway Deployment Setup

## Quick Deploy Steps

### 1. Connect Repository
Your GitHub repo is already connected to Railway.

### 2. Set Environment Variables

In Railway dashboard → Your Project → Variables, add:

```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

> **Note:** Use the same API keys from your local `backend/.env` file.

### 3. Add PostgreSQL (Required for Pixeltable)

1. Go to Railway dashboard
2. Click "New" → "Database" → "Add PostgreSQL"
3. Railway will automatically set `DATABASE_URL`

### 4. Deploy

Push your changes:
```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

Railway will automatically build and deploy.

## Configuration Files Created

- `railway.toml` - Railway deployment config
- `nixpacks.toml` - Build configuration (Python 3.11, FFmpeg, PostgreSQL)
- `Procfile` - Start command
- `backend/requirements.txt` - Python dependencies
- `backend/runtime.txt` - Python version

## Railway Settings

### Build Command (automatic)
```
cd backend && pip install -e .
```

### Start Command (automatic)
```
cd backend && python api.py
```

### Health Check
- Path: `/`
- Response: `{"status":"ok","message":"QuadRAG API is running"}`

## Frontend Configuration

### Connecting Streamlit UI to Railway Backend

The Streamlit frontend needs to know where your Railway backend is running.

**Option 1: Environment Variable (Recommended)**

Create a `.env` file in the `frontend/` directory:

```bash
cd frontend
echo "QUADRAG_API_URL=https://your-app.railway.app" > .env
```

Replace `https://your-app.railway.app` with your actual Railway backend URL.

**Option 2: Set Before Running**

```bash
export QUADRAG_API_URL=https://your-app.railway.app
streamlit run app.py
```

**Option 3: Local Development**

If running locally, the default `http://localhost:8000` will be used automatically.

### Finding Your Railway Backend URL

1. Go to Railway dashboard
2. Select your backend service
3. Click on "Settings" → "Domains"
4. Copy the Railway-provided domain (e.g., `https://your-app.railway.app`)

## Environment Variables Reference

### Backend (Railway)

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM | Yes |
| `OPENAI_API_KEY` | OpenAI API key for transcription/embeddings | Yes |
| `GOOGLE_API_KEY` | Google API key for Gemini embeddings | Yes |
| `PORT` | Server port (set by Railway) | Auto |
| `DATABASE_URL` | PostgreSQL URL (set by Railway) | Auto |

### Frontend (Local)

| Variable | Description | Required |
|----------|-------------|----------|
| `QUADRAG_API_URL` | Backend API URL (Railway or localhost) | No (defaults to localhost:8000) |

## Monitoring

- View logs in Railway dashboard → Deployments → View Logs
- Health check: `https://your-app.railway.app/`
- API docs: `https://your-app.railway.app/docs`

## Troubleshooting

### Frontend Connection Errors

**Error: "Connection refused" or "Cannot connect to backend"**

1. **Check Railway Backend URL:**
   - Verify your Railway backend is deployed and running
   - Get the correct URL from Railway dashboard → Settings → Domains
   - Ensure the URL starts with `https://` (not `http://`)

2. **Set Environment Variable:**
   ```bash
   # In frontend directory
   export QUADRAG_API_URL=https://your-app.railway.app
   # Or create .env file with:
   # QUADRAG_API_URL=https://your-app.railway.app
   ```

3. **Verify Backend Health:**
   - Visit `https://your-app.railway.app/health` in browser
   - Should return: `{"status":"ok","message":"QuadRAG API is running"}`

4. **Check Connection Status:**
   - The Streamlit UI shows connection status in the sidebar
   - Green badge = Connected
   - Red badge = Disconnected (check URL and backend status)

### Build Fails
- Check build logs for dependency errors
- Ensure `requirements.txt` is up to date

### App Crashes on Start
- Check environment variables are set
- View deployment logs for errors

### Pixeltable Database Issues
- Ensure PostgreSQL addon is added
- Pixeltable will auto-configure with `DATABASE_URL`

### Video Processing Slow
- Railway's free tier has limited resources
- Consider upgrading to Pro for more CPU/RAM
- Reduce `SPLIT_FRAMES_COUNT` to process fewer frames

## Local Development vs Railway

| Aspect | Local | Railway |
|--------|-------|---------|
| Port | 8000 | Dynamic (Railway sets PORT) |
| Database | Embedded PostgreSQL | Railway PostgreSQL |
| Video Storage | Local filesystem | Ephemeral (consider S3/Cloudinary) |
| FFmpeg | System install | Nixpacks installs |

## Important Notes

### Video Storage
Railway's filesystem is ephemeral. Uploaded videos will be lost on redeploy.

For production, consider:
1. **Cloudinary** - Video hosting service
2. **AWS S3** - Object storage
3. **Railway Volume** - Persistent storage addon

### Resource Limits
- Free tier: Limited CPU/RAM
- Pro tier: More resources, faster processing

### Cost
- Free tier: 500 hours/month, limited resources
- Pro tier: $5/month base + usage


