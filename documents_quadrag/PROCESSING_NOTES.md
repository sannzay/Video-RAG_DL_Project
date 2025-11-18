# Video Processing Notes

## Processing Time Expectations

### For a 5-minute video:
- **Video upload**: ~5 seconds
- **Video table creation**: ~5 seconds
- **Image Index** (45 frames): ~30-60 seconds
  - Frame extraction: ~5 seconds
  - CLIP embeddings: ~25-50 seconds
- **Audio Index** (200 chunks): **10-30 minutes**
  - Audio extraction: ~15 seconds
  - Chunk creation: ~5 seconds
  - **Transcription**: **10-25 minutes** (200 API calls to OpenAI)
  - Text extraction: ~5 seconds
  - Embeddings: ~2-5 minutes (processes as transcriptions complete)
- **Description Index** (45 frames): ~60-120 seconds
  - GPT-4o-mini descriptions: ~60-90 seconds
  - Embeddings: ~10-30 seconds

**Total Expected Time**: 15-35 minutes for a 5-minute video

### Why Transcription Takes So Long

1. **200 audio chunks** need to be transcribed
2. Each transcription is an **API call to OpenAI**
3. API calls are made **sequentially** (to avoid rate limits)
4. Each call takes **2-5 seconds**
5. **200 × 3 seconds = ~10 minutes minimum**

### Optimization Tips

1. **Reduce chunk count**: Increase `AUDIO_CHUNK_LENGTH` to 20 seconds (reduces chunks by 50%)
2. **Use shorter videos**: Process videos < 3 minutes for faster results
3. **Process in background**: The system processes incrementally, so you can continue using it

## Status Tracking

The system tracks:
- ✅ **Video uploaded**: File saved
- ⏳ **Processing**: Indexes being created
- ✅ **Completed**: At least one index created successfully
- ❌ **Failed**: All index creation failed

## Incremental Processing

Pixeltable processes computed columns **incrementally**:
- Transcription happens in background as you query
- Embeddings are created as transcriptions complete
- You can start querying before all processing is done

## Monitoring Progress

Check processing status:
```bash
curl http://localhost:8000/video/{video_id}/status
```

This shows:
- Current status (processing/completed/failed)
- Which indexes have been created
- Any error messages

## Troubleshooting

### Process seems stuck at transcription
- **Normal**: Transcription takes 10-30 minutes for long videos
- Check logs for progress messages
- Status endpoint will show when it completes

### Want faster processing?
- Reduce `AUDIO_CHUNK_LENGTH` in config (but reduces accuracy)
- Use shorter test videos
- Process multiple videos in parallel (different video_ids)

### API Rate Limits
- OpenAI has rate limits on transcription API
- If you hit limits, wait a few minutes and retry
- Consider upgrading OpenAI plan for higher limits

