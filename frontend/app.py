"""Streamlit UI for QuadRAG - Modern & Aesthetic Design."""

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
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "QuadRAG - A Four-Index Multimodal RAG System for Video Understanding"
    }
)

# API endpoint
API_BASE_URL = "http://localhost:8000"

# Modern CSS with gradient design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-bg: #0f0f23;
        --card-bg: #ffffff;
        --text-primary: #1a1a2e;
        --text-secondary: #6c757d;
        --border-color: #e9ecef;
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.15);
        --shadow-xl: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card Styles */
    .info-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .gradient-card {
        background: var(--primary-gradient);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: var(--shadow-xl);
        margin-bottom: 1.5rem;
    }
    
    /* Domain Context Box */
    .domain-context-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }
    
    /* Chat Messages */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1.2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 3rem;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-message {
        background: white;
        color: var(--text-primary);
        margin-right: 3rem;
        border: 1px solid var(--border-color);
        border-bottom-left-radius: 4px;
    }
    
    .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .message-content {
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Citations */
    .citation {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.75rem;
        padding: 0.75rem;
        background: #f8f9fa;
        border-left: 3px solid #667eea;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    
    .citation:hover {
        background: #e9ecef;
        transform: translateX(4px);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
    }
    
    .status-completed {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .status-failed {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    
    /* Video Card */
    .video-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .video-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    .video-card-active {
        border: 2px solid #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }
    
    /* Index Badge */
    .index-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
        background: #e9ecef;
        color: var(--text-primary);
    }
    
    .index-badge.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Stats Box */
    .stats-box {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stats-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: white;
    }
    
    /* Button Overrides */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* Input Overrides */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* Scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Info Section */
    .info-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }
    
    .info-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-text {
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: 0.9rem;
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .feature-item {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
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


def show_system_info():
    """Display system information and features."""
    st.markdown("""
    <div class="info-section">
        <div class="info-title">🎯 About QuadRAG</div>
        <div class="info-text">
            QuadRAG uses <strong>four parallel semantic indexes</strong> to enable rich, context-aware question answering about video content:
        </div>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-icon">🖼️</div>
                <div class="feature-label"><strong>Image Index</strong><br>Visual similarity search</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🎵</div>
                <div class="feature-label"><strong>Audio Index</strong><br>Transcribed dialogue search</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📝</div>
                <div class="feature-label"><strong>Description Index</strong><br>Scene understanding</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🎯</div>
                <div class="feature-label"><strong>Domain Index</strong><br>Context-specific analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_domain_context_dialog():
    """Show domain context input dialog."""
    st.markdown('<div class="main-header">🎬 QuadRAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Four-Index Multimodal RAG System for Video Understanding</div>', unsafe_allow_html=True)
    
    # Show system info
    show_system_info()
    
    st.markdown("---")
    
    st.markdown("""
    <div class="gradient-card">
        <h2 style="margin: 0 0 1rem 0; font-size: 1.8rem;">🎯 Set Domain Context</h2>
        <p style="margin: 0; font-size: 1.1rem; opacity: 0.95;">
            Specify the domain context to help QuadRAG focus on specific aspects of your videos.
            This creates a specialized index for domain-specific analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-section">
        <div class="info-title">💡 Example Domain Contexts</div>
        <div class="info-text">
            <strong>Emotions & Expressions:</strong> "Capture emotions and facial expressions"<br>
            <strong>Object Detection:</strong> "Identify objects and their locations"<br>
            <strong>Text Recognition:</strong> "Focus on text and written content"<br>
            <strong>Body Language:</strong> "Analyze body language and gestures"<br>
            <strong>Scene Analysis:</strong> "Describe scenes and environments"
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    domain_input = st.text_area(
        "Domain Context:",
        placeholder="e.g., Capture emotions and facial expressions in detail",
        height=120,
        help="Enter a description of what aspects of the video you want to focus on"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Set Context", type="primary", use_container_width=True):
            if domain_input.strip():
                st.session_state.domain_context = domain_input.strip()
                st.session_state.domain_set = True
                st.rerun()
            else:
                st.error("Please enter a domain context")
    
    with col2:
        if st.button("⏭️ Skip for now", use_container_width=True):
            st.session_state.domain_context = "General video analysis"
            st.session_state.domain_set = True
            st.rerun()


def upload_video_section():
    """Video upload section with modern design."""
    with st.expander("📤 Upload New Video", expanded=False):
        st.markdown("""
        <div class="info-text" style="margin-bottom: 1rem;">
            Upload a video file to start analyzing. Supported formats: MP4, AVI, MOV, MKV
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
            help="Select a video file from your device"
        )
        
        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📹 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
            
            if st.button("🚀 Upload and Process", type="primary", use_container_width=True):
                with st.spinner("📤 Uploading video..."):
                    # Upload video
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_BASE_URL}/upload-video", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        video_id = data["video_id"]
                        
                        st.success(f"✅ Video uploaded successfully!")
                        
                        # Start processing
                        with st.spinner("⚙️ Processing video (creating indexes)..."):
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
                                
                                st.info("🔄 Video is being processed. This may take a few minutes. You can check the status in the sidebar.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Failed to start processing")
                    else:
                        st.error("❌ Failed to upload video")


def show_video_library():
    """Show video library in sidebar with enhanced design."""
    st.sidebar.markdown("""
    <div style="margin-bottom: 1rem;">
        <h2 style="font-size: 1.5rem; font-weight: 700; margin: 0;">📚 Video Library</h2>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0.25rem 0 0 0;">
            Manage your uploaded videos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.uploaded_videos:
        st.sidebar.markdown("""
        <div class="info-card" style="text-align: center; padding: 2rem 1rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📹</div>
            <div style="color: #6c757d; font-size: 0.9rem;">No videos uploaded yet</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Stats
    total_videos = len(st.session_state.uploaded_videos)
    completed = sum(1 for v in st.session_state.uploaded_videos.values() if v.get("status") == "completed")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total", total_videos)
    with col2:
        st.metric("Ready", completed)
    
    st.sidebar.markdown("---")
    
    for video_id, video_info in st.session_state.uploaded_videos.items():
        # Get status from API
        try:
            status_response = requests.get(f"{API_BASE_URL}/video/{video_id}/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data["status"]
                indexes = status_data.get("indexes_created", [])
            else:
                status = video_info.get("status", "unknown")
                indexes = []
        except:
            status = video_info.get("status", "unknown")
            indexes = []
        
        # Update status in session state
        st.session_state.uploaded_videos[video_id]["status"] = status
        
        # Display video card
        is_active = video_id == st.session_state.active_video_id
        
        card_class = "video-card-active" if is_active else "video-card"
        
        st.sidebar.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                <strong style="font-size: 0.9rem;">📹 {video_info['filename']}</strong>
                {'⭐' if is_active else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status badge
        if status == "completed":
            st.sidebar.markdown(
                '<span class="status-badge status-completed">✓ Ready</span>',
                unsafe_allow_html=True
            )
        elif status == "processing":
            st.sidebar.markdown(
                '<span class="status-badge status-processing">⏳ Processing</span>',
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                '<span class="status-badge status-failed">✗ Failed</span>',
                unsafe_allow_html=True
            )
        
        # Indexes created
        if indexes:
            index_names = [idx.value if hasattr(idx, 'value') else str(idx) for idx in indexes]
            st.sidebar.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="font-size: 0.75rem; color: #6c757d; margin-bottom: 0.25rem;">Indexes:</div>
                <div>
                    {' '.join([f'<span class="index-badge active">{name}</span>' for name in index_names])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Select button
        if not is_active and status == "completed":
            if st.sidebar.button(f"Select Video", key=f"select_{video_id}", use_container_width=True):
                st.session_state.active_video_id = video_id
                st.rerun()
        
        st.sidebar.markdown("---")


def show_domain_context_panel():
    """Show current domain context with enhanced design."""
    with st.expander("🎯 Current Domain Context", expanded=False):
        st.markdown(
            f'<div class="domain-context-box">'
            f'<div style="font-size: 0.85rem; color: #6c757d; margin-bottom: 0.5rem; font-weight: 600;">DOMAIN CONTEXT</div>'
            f'<div style="font-size: 1.1rem; color: #1a1a2e; font-weight: 500;">{st.session_state.domain_context}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Change Domain Context", use_container_width=True):
                st.session_state.domain_context_editing = True
                st.rerun()
        
        if st.session_state.get("domain_context_editing", False):
            new_context = st.text_input("New domain context:", value=st.session_state.domain_context)
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Update", type="primary", use_container_width=True):
                    if new_context.strip():
                        st.session_state.domain_context = new_context.strip()
                        st.session_state.domain_context_editing = False
                        
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
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.domain_context_editing = False
                    st.rerun()


def show_chat_interface():
    """Show enhanced chat interface."""
    st.markdown("### 💬 Chat with Video")
    
    # Check if video is selected and processed
    if not st.session_state.active_video_id:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📹</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">
                No Video Selected
            </div>
            <div style="color: #6c757d;">
                Please upload and select a video to start chatting
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    video_info = st.session_state.uploaded_videos.get(st.session_state.active_video_id)
    if not video_info or video_info.get("status") != "completed":
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⏳</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">
                Video Processing
            </div>
            <div style="color: #6c757d;">
                Please wait for video processing to complete. This may take a few minutes.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Show active video info
    st.markdown(f"""
    <div class="info-card" style="padding: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">📹</span>
            <div>
                <div style="font-weight: 600; color: #1a1a2e;">Active Video</div>
                <div style="font-size: 0.9rem; color: #6c757d;">{video_info['filename']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<div class="message-header">You</div>'
                    f'<div class="message-content">{message["content"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message">'
                    f'<div class="message-header">🤖 QuadRAG Assistant</div>'
                    f'<div class="message-content">{message["content"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Show citations if available
                if "citations" in message and message["citations"]:
                    with st.expander(f"📎 View Citations ({len(message['citations'])})"):
                        for i, citation in enumerate(message["citations"], 1):
                            source = citation.get("source", "Unknown").upper()
                            timestamp = citation.get("timestamp", 0)
                            content = citation.get("content", "")[:200]
                            
                            st.markdown(
                                f'<div class="citation">'
                                f'<strong>[{i}]</strong> <span style="color: #667eea; font-weight: 600;">{source}</span> @ {timestamp:.1f}s<br>'
                                f'{content}...'
                                f'</div>',
                                unsafe_allow_html=True
                            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 3rem 2rem; margin-bottom: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">💬</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">
                Start a Conversation
            </div>
            <div style="color: #6c757d; margin-bottom: 1rem;">
                Ask questions about the video content. QuadRAG will search across all four indexes to provide comprehensive answers.
            </div>
            <div style="font-size: 0.9rem; color: #6c757d;">
                <strong>Example questions:</strong><br>
                "What emotions are shown in the video?"<br>
                "What objects appear in the scene?"<br>
                "What is being said in the audio?"
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
        with st.spinner("🤔 Thinking..."):
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
                        "content": data.get("answer", "No answer provided"),
                        "citations": data.get("citations", []),
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Sorry, I encountered an error processing your request. Please try again.",
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
    st.markdown('<div class="sub-header">A Four-Index Multimodal RAG System for Video Understanding</div>', unsafe_allow_html=True)
    
    # Sidebar
    show_video_library()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    with col1:
        show_domain_context_panel()
    with col2:
        show_system_info()
    
    upload_video_section()
    st.divider()
    show_chat_interface()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem;">QuadRAG v0.1.0</div>
        <div style="font-size: 0.85rem; color: #6c757d;">A Four-Index Multimodal RAG System</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
