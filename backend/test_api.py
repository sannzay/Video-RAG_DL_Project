"""Simple test script for QuadRAG API."""

import time

import requests

API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test API health check."""
    print("Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✅ Health check passed\n")


def test_upload_video(video_path: str):
    """Test video upload."""
    print(f"Testing video upload: {video_path}")
    
    with open(video_path, "rb") as f:
        files = {"file": (video_path, f)}
        response = requests.post(f"{API_BASE_URL}/upload-video", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Video ID: {data['video_id']}")
        print(f"File path: {data['file_path']}")
        print("✅ Video upload passed\n")
        return data["video_id"]
    else:
        print(f"❌ Video upload failed: {response.text}\n")
        return None


def test_process_video(video_id: str):
    """Test video processing."""
    print(f"Testing video processing for: {video_id}")
    
    response = requests.post(
        f"{API_BASE_URL}/process-video",
        json={"video_id": video_id}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Processing status: {data['status']}")
        print("✅ Video processing started\n")
        return True
    else:
        print(f"❌ Video processing failed: {response.text}\n")
        return False


def test_check_status(video_id: str, wait_for_completion: bool = True):
    """Test status checking."""
    print(f"Checking status for: {video_id}")
    
    max_attempts = 60  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        response = requests.get(f"{API_BASE_URL}/video/{video_id}/status")
        
        if response.status_code == 200:
            data = response.json()
            status = data["status"]
            indexes = data["indexes_created"]
            
            print(f"Status: {status}, Indexes: {indexes}")
            
            if status == "completed":
                print("✅ Video processing completed\n")
                return True
            elif status == "failed":
                print(f"❌ Video processing failed: {data.get('error_message')}\n")
                return False
            elif wait_for_completion:
                print(f"Waiting... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(5)
                attempt += 1
            else:
                return False
        else:
            print(f"❌ Status check failed: {response.text}\n")
            return False
    
    print("❌ Timeout waiting for processing\n")
    return False




def test_chat(video_id: str, session_id: str, query: str, domain_context: str = None):
    """Test chat functionality."""
    print(f"Testing chat with query: {query}")
    
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "session_id": session_id,
            "video_id": video_id,
            "query": query,
            "domain_context": domain_context,
        }
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"\nAnswer: {data['answer']}")
        print(f"\nCitations ({len(data['citations'])}):")
        for i, citation in enumerate(data['citations'][:3], 1):  # Show first 3
            print(f"  [{i}] {citation['source']} @ {citation['timestamp']:.1f}s")
            print(f"      {citation['content'][:100]}...")
        print(f"\nProcessing time: {data['processing_time']:.2f}s")
        print("✅ Chat test passed\n")
        return True
    else:
        print(f"❌ Chat failed: {response.text}\n")
        return False


def test_reprocess_video(video_id: str):
    """Test video re-processing."""
    print(f"Testing video re-processing for: {video_id}")

    response = requests.post(
        f"{API_BASE_URL}/reprocess-video",
        json={"video_id": video_id}
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Message: {data['message']}")
        print("✅ Video re-processing started\n")
        return True
    else:
        print(f"❌ Video re-processing failed: {response.text}\n")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("QuadRAG API Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Health check
    try:
        test_health_check()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("\n⚠️ Make sure the backend server is running!")
        print("   Run: cd backend && python api.py\n")
        return
    
    # For remaining tests, you need to provide a video file
    print("=" * 60)
    print("To test video upload and processing:")
    print("1. Make sure you have a video file")
    print("2. Edit this script and set VIDEO_PATH")
    print("3. Make sure you have API keys configured in .env")
    print("=" * 60 + "\n")
    
    # Example:
    # VIDEO_PATH = "path/to/your/video.mp4"
    # video_id = test_upload_video(VIDEO_PATH)
    # if video_id:
    #     test_process_video(video_id)
    #     test_check_status(video_id, wait_for_completion=True)
    #     
    #     session_id = "test_session_123"
    #     test_chat(video_id, session_id, "What happens in the video?", domain_context)


if __name__ == "__main__":
    main()


