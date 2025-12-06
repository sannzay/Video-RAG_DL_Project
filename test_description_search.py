#!/usr/bin/env python3
"""
Test script to verify Description Index search functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from quadrag.retrieval.search_engine import VideoSearchEngine
from quadrag.video.registry import get_all_videos

def test_description_search():
    """Test description search functionality."""
    print("Testing Description Index Search...")

    # Get the first processed video with description view
    videos = get_all_videos()
    test_video = None

    for video_id, video_info in videos.items():
        if video_info.description_view_name:
            test_video = video_info
            break

    if not test_video:
        print("❌ No video found with description view")
        print("Available videos:")
        for video_id, video_info in videos.items():
            print(f"  {video_id}: description_view={video_info.description_view_name}")
        return False

    print(f"Found video with description index: {test_video.video_id}")

    # Create search engine
    try:
        search_engine = VideoSearchEngine(test_video.video_id)
        print("✅ Search engine created successfully")
    except Exception as e:
        print(f"❌ Failed to create search engine: {e}")
        return False

    # Test description search
    try:
        results = search_engine.search_description_index("test query", top_k=3)
        print(f"✅ Description search completed, found {len(results)} results")

        if results:
            print("Sample result:")
            print(f"  Content: {results[0].content[:100]}...")
            print(f"  Timestamp: {results[0].timestamp}")
            print(f"  Similarity: {results[0].similarity}")
            print(f"  Source: {results[0].source}")

        return True

    except Exception as e:
        print(f"❌ Description search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_description_search()
    sys.exit(0 if success else 1)
