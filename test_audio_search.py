#!/usr/bin/env python3
"""
Test script to verify audio search functionality.
Run this locally to test if transcriptions are being stored and retrieved properly.
"""

import sys
import os
sys.path.insert(0, 'backend')

def test_audio_search():
    """Test if audio search can retrieve transcriptions."""
    try:
        # Import required modules
        from quadrag.video.registry import get_all_videos
        from quadrag.retrieval.search_engine import SearchEngine
        import logging

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Get all registered videos
        videos = get_all_videos()
        logger.info(f"Found {len(videos)} registered videos")

        if not videos:
            logger.error("No videos registered. Please upload and process a video first.")
            return False

        # Test each video
        for video_id, video_info in videos.items():
            logger.info(f"\nTesting video: {video_id}")
            logger.info(f"  Audio view: {video_info.audio_view_name}")
            logger.info(f"  Video table: {video_info.video_table_name}")

            # Create search engine
            search_engine = SearchEngine(video_id)

            # Test audio search
            try:
                results = search_engine.search_audio_index("test query")
                logger.info(f"  Audio search returned {len(results)} results")

                if results:
                    logger.info("  ✅ Audio search working!")
                    for i, result in enumerate(results[:3]):  # Show first 3 results
                        logger.info(f"    Result {i+1}: '{result.content[:100]}...' at {result.timestamp:.1f}s")
                    return True
                else:
                    logger.warning("  ⚠️ Audio search returned no results")

            except Exception as e:
                logger.error(f"  ❌ Audio search failed: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")

        return False

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Testing audio search functionality...")
    success = test_audio_search()
    if success:
        print("✅ Audio search test PASSED")
    else:
        print("❌ Audio search test FAILED")
    sys.exit(0 if success else 1)
