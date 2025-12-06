#!/usr/bin/env python3
"""Test script to debug audio transcription issues."""

import os
import sys

# Add backend to path
sys.path.insert(0, 'backend/src')

def test_transcription():
    """Test transcription computation."""
    try:
        import pixeltable as pxt
        from quadrag.config import get_settings
        from quadrag.video.registry import get_video_from_registry

        settings = get_settings()
        print(f"OPENAI_API_KEY set: {bool(settings.OPENAI_API_KEY)}")
        print(f"GROQ_API_KEY set: {bool(settings.GROQ_API_KEY)}")

        # Initialize Pixeltable
        pxt.init()

        # Get video info
        video_id = "d8ac7c2b-80a5-4808-b26f-154f3479a157"
        video_info = get_video_from_registry(video_id)

        if not video_info:
            print(f"Video {video_id} not found")
            return

        print(f"Video found: {video_info.video_id}")
        print(f"Audio view name: {video_info.audio_view_name}")

        # Try to access audio view
        try:
            audio_view = video_info.audio_view
            print("Audio view accessible")

            # Check if transcription column exists
            try:
                _ = audio_view.transcription
                print("✓ Transcription column exists")
            except AttributeError:
                print("✗ Transcription column missing")

            try:
                _ = audio_view.transcript_text
                print("✓ Transcript text column exists")
            except AttributeError:
                print("✗ Transcript text column missing")

            # Try to collect a few chunks
            print("Collecting audio chunks...")
            chunks = audio_view.select(
                audio_view.start_time_sec,
                audio_view.end_time_sec,
                audio_view.transcription,
                audio_view.transcript_text,
            ).limit(3).collect()

            print(f"Collected {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                raw_transcript = chunk.get("transcription", "")
                text_transcript = chunk.get("transcript_text", "")
                start_time = chunk.get("start_time_sec", 0)

                print(f"\nChunk {i} (start: {start_time}s):")
                print(f"  Raw transcription: {str(raw_transcript)[:200]}...")
                print(f"  Text transcription: '{str(text_transcript)[:200]}'...")

            except Exception as e:
            print(f"Error accessing audio view: {e}")
                import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transcription()