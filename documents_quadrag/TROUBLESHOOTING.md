# QuadRAG Troubleshooting Guide

## Common Issues and Solutions

### Issue: Video Processing Stuck at Transcription

**Symptoms:**
- Process hangs at "Adding audio transcription"
- No progress for 30+ minutes
- Logs show transcription started but never completes

**Root Cause:**
- Pixeltable's `add_computed_column` may try to validate the column
- For 200 audio chunks, this would trigger 200 API calls synchronously
- Each API call takes 2-5 seconds = 10-30 minutes total

**Solution (Already Implemented):**
- Transcription columns are now added without forcing evaluation
- Transcriptions compute lazily (on-demand) when you query
- Audio search uses text matching instead of embedding index (faster)

**What to Do:**
1. **Restart the backend** - The fix is already in place
2. **Upload a new video** - Old videos may have partial state
3. **Wait for processing** - It should complete in < 1 minute now
4. **Check status** - Video should show as "completed" quickly

### Issue: Audio Search Returns No Results

**Symptoms:**
- Query returns empty results from audio index
- Logs show "No matches found in first 50 chunks"

**Root Cause:**
- Text search only checks first 50 chunks (to avoid blocking)
- Transcriptions may not be computed yet for those chunks

**Solution:**
- Wait a few minutes for transcriptions to compute
- Try querying again - transcriptions compute on-demand
- Use more specific query terms

### Issue: Processing Takes Too Long

**Expected Times:**
- **Short video (< 2 min)**: 2-5 minutes total
- **Medium video (2-5 min)**: 5-15 minutes total  
- **Long video (> 5 min)**: 15-30 minutes total

**Optimization Tips:**
1. **Reduce chunk count**: Increase `AUDIO_CHUNK_LENGTH` to 20 seconds
2. **Use shorter videos**: Test with < 2 minute videos first
3. **Process in background**: System continues processing while you use UI

### Issue: "Video not found" Error

**Symptoms:**
- Status shows video not found
- Chat returns 404 error

**Solution:**
1. Check video was uploaded successfully
2. Check video processing completed (status = "completed")
3. Verify video_id matches between upload and query

### Issue: Domain Index Not Working

**Symptoms:**
- Domain context set but no domain results
- Domain index shows as created but search fails

**Solution:**
1. Make sure Image Index was created first (domain depends on frames)
2. Set domain context AFTER video processing completes
3. Wait a moment for domain captions to generate

### Issue: API Timeout Errors

**Symptoms:**
- OpenAI API timeout errors
- Rate limit errors

**Solution:**
1. Check API key is valid and has credits
2. Wait a few minutes and retry
3. Reduce `AUDIO_CHUNK_LENGTH` to process fewer chunks
4. Use shorter videos for testing

## Performance Optimization

### Reduce Processing Time

Edit `backend/.env`:
```bash
# Increase chunk size (fewer chunks = faster)
AUDIO_CHUNK_LENGTH=20  # Default: 10

# Reduce frame count (fewer frames = faster)
SPLIT_FRAMES_COUNT=30  # Default: 45
```

### Improve Search Quality

1. **Wait for transcriptions**: Let transcriptions compute before querying
2. **Use specific queries**: More specific queries work better with text search
3. **Set domain context**: Domain-specific queries work better

## Debugging

### Check Processing Status

```bash
curl http://localhost:8000/video/{video_id}/status
```

### Check Logs

Backend logs show:
- Which indexes were created
- Any errors during processing
- Search query processing

### Test Individual Components

1. **Test Image Index**: Query with image-related questions
2. **Test Audio Index**: Query with speech-related questions  
3. **Test Description Index**: Query with scene-related questions
4. **Test Domain Index**: Query with domain-specific questions

## Getting Help

If issues persist:
1. Check backend logs for error messages
2. Verify API keys are correct
3. Check video file format (MP4, AVI, MOV supported)
4. Ensure FFmpeg is installed
5. Try with a shorter test video first

