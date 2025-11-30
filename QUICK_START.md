# 🚀 QuadRAG Quick Start

## ✅ Current Status

**Backend:** ✅ Running at http://localhost:8000
**Frontend:** Starting...

## 📍 Access Points

- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Frontend UI:** http://localhost:8501 (opening in browser...)

## 🎯 Next Steps

### If Frontend is Not Running:

Open a **new terminal** and run:

```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag/frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or use the startup script:
```bash
cd /Users/sanju/Documents/2ndSemGSU/DeepLearning/Project/QuadRag
./start_frontend.sh
```

## 🎬 Using QuadRAG

1. **Open Browser:** Go to http://localhost:8501
2. **Upload Video:** Click "Upload Video" and select your video file
3. **Wait for Processing:** Status will show "Processing" then "Completed"
4. **Set Domain Context (Optional):** Enter domain-specific instructions
5. **Start Chatting:** Ask questions about your video!

## 🔍 Verify Backend is Working

Test the API:
```bash
curl http://localhost:8000/
```

Should return: `{"status":"ok","message":"QuadRAG API is running"}`

## 🛑 Stopping Servers

Press `Ctrl+C` in the terminal where each server is running.

## 📝 Notes

- Backend processes videos in the background
- Video processing can take 1-5 minutes depending on video length
- Transcriptions compute lazily (on-demand when you query)
- Check backend terminal for detailed processing logs

