"""Streamlit UI for QuadRAG."""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import requests
import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="QuadRAG - Video Understanding",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API endpoint
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .domain-context-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .citation {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #fff;
        border-left: 3px solid #1f77b4;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-completed {
        background-color: #d4edda;
        color: #155724;
    }
    .status-processing {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "domain_context" not in st.session_state:
        st.session_state.domain_context = None
    
    if "domain_set" not in st.session_state:
        st.session_state.domain_set = False
    
    if "active_video_id" not in st.session_state:
        st.session_state.active_video_id = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "uploaded_videos" not in st.session_state:
        st.session_state.uploaded_videos = {}


def show_domain_context_dialog():
    """Show domain context input dialog."""
    st.markdown('<div class="main-header">🎬 QuadRAG - Video Understanding</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Four-Index Multimodal RAG System</div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Set Domain Context")
    st.markdown("""
    Welcome to QuadRAG! To get started, please specify the domain context for analyzing videos.
    This helps the system focus on specific aspects of the video content.
    
    **Examples:**
    - "Capture emotions and facial expressions"
    - "Identify objects and their locations"
    - "Focus on text and written content"
    - "Analyze body language and gestures"
    """)
    
    domain_input = st.text_area(
        "Domain Context:",
        placeholder="e.g., Capture emotions and facial expressions",
        height=100,
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Set Context", type="primary"):
            if domain_input.strip():
                st.session_state.domain_context = domain_input.strip()
                st.session_state.domain_set = True
                st.rerun()
            else:
                st.error("Please enter a domain context")
    
    with col2:
        if st.button("Skip for now"):
            st.session_state.domain_context = "General video analysis"
            st.session_state.domain_set = True
            st.rerun()


def upload_video_section():
    """Video upload section."""
    with st.expander("📤 Upload New Video", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )
        
        if uploaded_file is not None:
            if st.button("Upload and Process"):
                with st.spinner("Uploading video..."):
                    # Upload video
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_BASE_URL}/upload-video", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        video_id = data["video_id"]
                        
                        st.success(f"✅ Video uploaded: {uploaded_file.name}")
                        
                        # Start processing
                        with st.spinner("Processing video (creating indexes)..."):
                            process_response = requests.post(
                                f"{API_BASE_URL}/process-video",
                                json={"video_id": video_id}
                            )
                            
                            if process_response.status_code == 200:
                                st.session_state.uploaded_videos[video_id] = {
                                    "filename": uploaded_file.name,
                                    "upload_time": datetime.now(),
                                    "status": "processing",
                                }
                                st.session_state.active_video_id = video_id
                                
                                st.info("🔄 Video is being processed. This may take a few minutes...")
                                st.rerun()
                            else:
                                st.error("Failed to start processing")
                    else:
                        st.error("Failed to upload video")


def show_video_library():
    """Show video library in sidebar."""
    st.sidebar.markdown("## 📚 Video Library")
    
    if not st.session_state.uploaded_videos:
        st.sidebar.info("No videos uploaded yet")
        return
    
    for video_id, video_info in st.session_state.uploaded_videos.items():
        # Get status from API
        try:
            status_response = requests.get(f"{API_BASE_URL}/video/{video_id}/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data["status"]
                indexes = status_data["indexes_created"]
            else:
                status = video_info["status"]
                indexes = []
        except:
            status = video_info["status"]
            indexes = []
        
        # Update status in session state
        st.session_state.uploaded_videos[video_id]["status"] = status
        
        # Display video card
        is_active = video_id == st.session_state.active_video_id
        
        with st.sidebar.container():
            if is_active:
                st.markdown("**📹 " + video_info["filename"] + "** ⭐")
            else:
                st.markdown("📹 " + video_info["filename"])
            
            # Status badge
            if status == "completed":
                st.markdown(
                    '<span class="status-badge status-completed">✓ Ready</span>',
                    unsafe_allow_html=True
                )
            elif status == "processing":
                st.markdown(
                    '<span class="status-badge status-processing">⏳ Processing</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="status-badge status-failed">✗ Failed</span>',
                    unsafe_allow_html=True
                )
            
            # Indexes created
            if indexes:
                st.caption(f"Indexes: {', '.join([idx.value for idx in indexes])}")
            
            # Select button
            if not is_active and status == "completed":
                if st.sidebar.button(f"Select", key=f"select_{video_id}"):
                    st.session_state.active_video_id = video_id
                    st.rerun()
            
            st.sidebar.divider()


def show_domain_context_panel():
    """Show current domain context."""
    with st.expander("🎯 Current Domain Context", expanded=False):
        st.markdown(
            f'<div class="domain-context-box">'
            f'<strong>Domain:</strong> {st.session_state.domain_context}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if st.button("Change Domain Context"):
            new_context = st.text_input("New domain context:")
            if st.button("Update") and new_context.strip():
                st.session_state.domain_context = new_context.strip()
                
                # Update domain index for active video
                if st.session_state.active_video_id:
                    with st.spinner("Updating domain index..."):
                        response = requests.post(
                            f"{API_BASE_URL}/set-domain-context",
                            json={
                                "session_id": st.session_state.session_id,
                                "video_id": st.session_state.active_video_id,
                                "domain_context": st.session_state.domain_context,
                            }
                        )
                        if response.status_code == 200:
                            st.success("✅ Domain context updated!")
                        else:
                            st.error("Failed to update domain context")
                st.rerun()


def show_chat_interface():
    """Show chat interface."""
    st.markdown("### 💬 Chat with Video")
    
    # Check if video is selected and processed
    if not st.session_state.active_video_id:
        st.info("👆 Please upload and select a video to start chatting")
        return
    
    video_info = st.session_state.uploaded_videos.get(st.session_state.active_video_id)
    if not video_info or video_info["status"] != "completed":
        st.warning("⏳ Please wait for video processing to complete")
        return
    
    # Show active video
    st.caption(f"Active video: **{video_info['filename']}**")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">'
                f'<strong>You:</strong><br>{message["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message">'
                f'<strong>Assistant:</strong><br>{message["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Show citations if available
            if "citations" in message and message["citations"]:
                with st.expander("📎 View Citations"):
                    for i, citation in enumerate(message["citations"], 1):
                        st.markdown(
                            f'<div class="citation">'
                            f'<strong>[{i}]</strong> {citation["source"].upper()} @ {citation["timestamp"]:.1f}s<br>'
                            f'{citation["content"][:200]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )
    
    # Chat input
    user_query = st.chat_input("Ask a question about the video...")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Set domain context if not already set
        if st.session_state.domain_context:
            try:
                requests.post(
                    f"{API_BASE_URL}/set-domain-context",
                    json={
                        "session_id": st.session_state.session_id,
                        "video_id": st.session_state.active_video_id,
                        "domain_context": st.session_state.domain_context,
                    }
                )
            except:
                pass  # Domain index may already exist
        
        # Call chat API
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "video_id": st.session_state.active_video_id,
                        "query": user_query,
                        "domain_context": st.session_state.domain_context,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "citations": data["citations"],
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Sorry, I encountered an error processing your request.",
                    })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                })
        
        st.rerun()


def main():
    """Main application."""
    initialize_session_state()
    
    # Show domain context dialog if not set
    if not st.session_state.domain_set:
        show_domain_context_dialog()
        return
    
    # Main interface
    st.markdown('<div class="main-header">🎬 QuadRAG</div>', unsafe_allow_html=True)
    
    # Sidebar
    show_video_library()
    
    # Main content
    show_domain_context_panel()
    upload_video_section()
    st.divider()
    show_chat_interface()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**QuadRAG v0.1.0**")
    st.sidebar.caption("A Four-Index Multimodal RAG System")


if __name__ == "__main__":
    main()


