"""Streamlit UI for QuadRAG - Modern & Aesthetic Design."""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

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

# API endpoint - Fixed Railway backend URL
# Can be overridden with QUADRAG_API_URL environment variable for local development
RAILWAY_BACKEND_URL = "https://video-ragdlproject-production.up.railway.app"

def get_api_base_url() -> str:
    """Get API base URL from environment variable or use Railway default."""
    # Check environment variable first (for local development override)
    env_url = os.getenv("QUADRAG_API_URL")
    if env_url:
        return env_url
    # Default to Railway backend
    return RAILWAY_BACKEND_URL

# Initialize API_BASE_URL
API_BASE_URL = get_api_base_url()

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


def check_api_connection() -> tuple:
    """Check if the API backend is accessible."""
    # Get current API URL (may have been updated)
    current_url = get_api_base_url()
    
    # Try health endpoint first, then root endpoint
    endpoints_to_try = ["/health", "/"]
    
    for endpoint in endpoints_to_try:
        try:
            # Disable SSL verification warnings for Railway (they use valid certs)
            response = requests.get(
                f"{current_url}{endpoint}", 
                timeout=10,
                verify=True  # Keep SSL verification enabled
            )
            if response.status_code == 200:
                return True, "Connected"
            else:
                # If we get a response (even non-200), the server is reachable
                return False, f"Backend returned status {response.status_code}"
        except requests.exceptions.SSLError as e:
            return False, f"SSL Error: {str(e)[:100]}"
        except requests.exceptions.ConnectionError as e:
            # Check if it's a specific connection error
            error_str = str(e).lower()
            if "name resolution" in error_str or "nodename nor servname" in error_str:
                return False, "DNS resolution failed - Check URL"
            elif "refused" in error_str:
                return False, "Connection refused - Backend may be down"
            else:
                return False, f"Connection error: {str(e)[:100]}"
        except requests.exceptions.Timeout:
            # Only report timeout if all endpoints fail
            if endpoint == endpoints_to_try[-1]:
                return False, "Connection timeout - Backend not responding"
            continue
        except Exception as e:
            # For other errors, return the error message
            return False, f"Error: {str(e)[:100]}"
    
    # If we get here, all endpoints failed
    return False, "Backend not reachable"


def show_connection_status():
    """Display API connection status in the sidebar."""
    # Get current API URL
    current_api_url = get_api_base_url()
    
    # Update global API_BASE_URL
    global API_BASE_URL
    API_BASE_URL = current_api_url
    
    is_connected, message = check_api_connection()
    
    if is_connected:
        st.sidebar.success("Backend Connected")
    else:
        st.sidebar.error("Backend Disconnected")

        with st.sidebar.expander("🔧 Connection Details"):
            st.write(f"**URL:** {current_api_url}")
            st.write(f"**Status:** {message}")
            if st.button("🔄 Retry", use_container_width=True):
                st.rerun()


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
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 About")
    st.sidebar.caption("Four indexes: **Image** • **Audio** • **Description** • **Domain**")


def show_domain_context_dialog():
    """Show domain context input dialog."""
    # Show system info
    show_system_info()

    st.markdown("---")
    
    st.markdown("""
    <div class="gradient-card">
        <h2 style="margin: 0 0 1rem 0; font-size: 1.8rem;">🎯 Set Domain Context</h2>
        <p style="margin: 0; font-size: 1.1rem; opacity: 0.95;">
            Specify what aspects of videos you want to focus on for specialized analysis.
        </p>
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
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 2rem; 
                border-radius: 16px; 
                margin: 1.5rem 0;
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
        <h2 style="margin: 0 0 1rem 0; font-size: 1.5rem; color: white;">📤 Upload Video</h2>
        <p style="margin: 0; opacity: 0.9; font-size: 0.95rem;">Upload a <strong>.mp4</strong> video file to start analyzing</p>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 0.85rem;">⚠️ Only MP4 format is supported</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose MP4 video file",
        type=["mp4"],
        key="video_uploader",
        label_visibility="collapsed",
        help="Only MP4 video files are supported"
    )
    
    return uploaded_file


def get_required_indexes(has_domain_context: bool = False) -> list:
    """Get list of required indexes based on whether domain context is provided.
    
    Args:
        has_domain_context: Whether domain context was provided
        
    Returns:
        List of required index names
    """
    base_indexes = ["AUDIO", "IMAGE", "DESCRIPTION"]
    if has_domain_context:
        return base_indexes + ["DOMAIN"]
    return base_indexes


def show_video_library():
    """Show video library in sidebar with enhanced design."""
    st.sidebar.markdown("### 📚 Videos")
    
    if not st.session_state.uploaded_videos:
        st.sidebar.info("No videos uploaded yet")
        return
    
    # Check if domain context is set (for determining required indexes)
    has_domain_context = bool(st.session_state.get("domain_context") and 
                              st.session_state.domain_context != "General video analysis")
    
    # Stats - only count as completed if all required indexes are ready
    total_videos = len(st.session_state.uploaded_videos)
    required_indexes = get_required_indexes(has_domain_context)
    
    completed = 0
    processing = 0
    
    for v in st.session_state.uploaded_videos.values():
        indexes = v.get("indexes", [])
        index_names = [idx.value if hasattr(idx, 'value') else str(idx).upper() for idx in indexes]
        has_required_indexes = all(idx in index_names for idx in required_indexes)
        
        if v.get("status") == "completed" and has_required_indexes:
            completed += 1
        else:
            processing += 1

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.metric("Total", total_videos)
    with col2:
        st.metric("Ready", completed)
    with col3:
        st.metric("Processing", processing)

    # Check if any videos are processing
    has_processing = any(v.get("status") == "processing" for v in st.session_state.uploaded_videos.values())
    
    # Manual refresh button - make it more prominent when processing
    if has_processing:
        if st.sidebar.button("🔄 Refresh Status", help="Click to check if video processing is complete", type="primary", use_container_width=True):
            st.rerun()
        st.sidebar.caption("💡 Click Refresh Status to check processing progress")
    else:
        if st.sidebar.button("🔄 Refresh Status", help="Manually refresh video processing status"):
            st.rerun()
    
    st.sidebar.markdown("---")
    
    for video_id, video_info in st.session_state.uploaded_videos.items():
        # Initialize variables
        index_errors = {}

        # Get status from API
        try:
            current_url = get_api_base_url()
            status_response = requests.get(f"{current_url}/video/{video_id}/status", timeout=5)
            if status_response.status_code == 200:
                status_data = status_response.json()
                api_status = status_data["status"]
                indexes = status_data.get("indexes_created", [])
                index_errors = status_data.get("index_errors", {})

                # Convert indexes to strings for comparison
                index_names = [idx.value if hasattr(idx, 'value') else str(idx).upper() for idx in indexes]
                
                # Determine required indexes based on whether domain context was provided
                # Check if domain context exists and is not the default
                has_domain_context = bool(st.session_state.get("domain_context") and 
                                        st.session_state.domain_context != "General video analysis")
                required_indexes = get_required_indexes(has_domain_context)
                
                # Check if all required indexes are present
                has_required_indexes = all(idx in index_names for idx in required_indexes)
                
                # Mark as completed if API says completed, even if some indexes failed
                # The video is still usable with available indexes
                if api_status == "completed":
                    status = "completed"
                else:
                    status = api_status
                
                # Debug logging
                if status != video_info.get("status"):
                    st.sidebar.write(f"🔄 Status change for {video_id[:8]}...: {video_info.get('status')} → {status}")
            else:
                status = video_info.get("status", "unknown")
                indexes = []
        except requests.exceptions.RequestException as e:
            # If connection fails, keep last known status
            status = video_info.get("status", "unknown")
            indexes = []
            st.sidebar.write(f"⚠️ Status check failed for {video_id[:8]}...: {str(e)}")
        
        # Update status in session state
        old_status = st.session_state.uploaded_videos[video_id].get("status")
        st.session_state.uploaded_videos[video_id]["status"] = status
        st.session_state.uploaded_videos[video_id]["indexes"] = indexes

        # Trigger UI refresh when status changes from processing to completed
        if old_status == "processing" and status == "completed":
            st.success(f"✅ Video '{video_info['filename']}' processing completed!")
            st.rerun()
        
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

        # Determine required indexes for this video
        has_domain_context = bool(st.session_state.get("domain_context") and 
                                 st.session_state.domain_context != "General video analysis")
        required_indexes = get_required_indexes(has_domain_context)
        all_possible_indexes = ["AUDIO", "IMAGE", "DESCRIPTION", "DOMAIN"]
        index_names = [idx.value if hasattr(idx, 'value') else str(idx).upper() for idx in indexes]
        
        # Status badge - only show Ready if all required indexes are present
        if status == "completed":
            has_required_indexes = all(idx in index_names for idx in required_indexes)

            if has_required_indexes:
                st.sidebar.markdown(
                    '<span class="status-badge status-completed">✓ Ready</span>',
                    unsafe_allow_html=True
                )
            else:
                # Show ready with warning about missing indexes
                st.sidebar.markdown(
                    '<span class="status-badge status-completed" style="background: #ffc107; color: #000;">⚠️ Ready (Partial)</span>',
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
            
        # Indexes created - show all possible indexes with status (for all statuses)
        st.sidebar.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="font-size: 0.75rem; color: #6c757d; margin-bottom: 0.25rem;">Indexes:</div>
            <div>
                {' '.join([
                    f'<span class="index-badge active" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white;">{name}</span>' 
                    if name in index_names 
                    else f'<span class="index-badge" style="background: {"#e9ecef" if name in required_indexes else "#f8f9fa"}; color: #6c757d; opacity: {"0.6" if name in required_indexes else "0.4"}; border: {"1px solid #dee2e6" if name in required_indexes else "1px dashed #dee2e6"};">{name}</span>'
                    for name in all_possible_indexes
                ])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show index errors if any
        has_errors = index_errors and any(idx in index_errors for idx in all_possible_indexes)
        has_missing_indexes = not all(idx in index_names for idx in required_indexes)

        if has_errors:
            st.sidebar.markdown("**Index Errors:**")
            for idx_name in all_possible_indexes:
                if idx_name in index_errors:
                    with st.sidebar.expander(f"❌ {idx_name} Error", expanded=False):
                        st.error(index_errors[idx_name][:500] + ("..." if len(index_errors[idx_name]) > 500 else ""))

        # Re-process button for failed indexes or missing required indexes
        if has_errors or (status == "completed" and has_missing_indexes):
            missing_list = [idx for idx in required_indexes if idx not in index_names] if has_missing_indexes else []
            error_list = list(index_errors.keys()) if has_errors else []

            all_missing = list(set(missing_list + error_list))

            if all_missing:
                button_text = f"🔄 Re-process ({', '.join(all_missing)})"
                button_help = f"Retry creating failed indexes: {', '.join(all_missing)}"
            else:
                button_text = "🔄 Re-process Video"
                button_help = "Retry creating failed or missing indexes"

            if st.sidebar.button(button_text, key=f"reprocess_{video_id}", help=button_help):
                try:
                    current_url = get_api_base_url()
                    process_payload = {"video_id": video_id}
                    if st.session_state.get("domain_context"):
                        process_payload["domain_context"] = st.session_state.domain_context
                        process_payload["session_id"] = st.session_state.session_id

                    response = requests.post(
                        f"{current_url}/reprocess-video",
                        json=process_payload,
                        timeout=30
                    )
                    if response.status_code == 200:
                        st.sidebar.success("🔄 Re-processing started!")
                        st.rerun()
                    else:
                        st.sidebar.error(f"Failed to start re-processing: {response.status_code}")
                except Exception as e:
                    st.sidebar.error(f"Re-processing failed: {str(e)}")

        # Select button - only show if there are multiple videos and this one is not active
        # If there's only one video, it's automatically selected
        total_videos_count = len(st.session_state.uploaded_videos)
        if not is_active and status == "completed" and total_videos_count > 1:
            has_required_indexes = all(idx in index_names for idx in required_indexes)

            if has_required_indexes:
                button_text = "Select Video"
            else:
                button_text = "Select Video (Partial)"

            if st.sidebar.button(button_text, key=f"select_{video_id}", use_container_width=True):
                st.session_state.active_video_id = video_id
                st.rerun()
            
        st.sidebar.markdown("---")


def show_domain_context_panel():
    """Show current domain context with enhanced design."""
    # Stylish domain context card
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; 
                padding: 1.5rem; 
                border-radius: 16px; 
                margin: 1rem 0;
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
                border-left: 4px solid rgba(255,255,255,0.5);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1.2rem; color: white; font-weight: 600;">🎯 Domain Context</h3>
        </div>
        <div style="font-size: 1rem; opacity: 0.95; line-height: 1.6;" id="domain-context-display">
            {context}
        </div>
    </div>
    """.format(context=st.session_state.domain_context), unsafe_allow_html=True)
    
    # Edit button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("✏️ Edit", use_container_width=True, type="secondary"):
            st.session_state.domain_context_editing = True
            st.rerun()
    
    # Editing mode
    if st.session_state.get("domain_context_editing", False):
        st.markdown("---")
        new_context = st.text_area(
            "Update Domain Context:",
            value=st.session_state.domain_context,
            height=100,
            help="Specify what aspects of videos you want to focus on"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Save", type="primary", use_container_width=True):
                if new_context.strip():
                    st.session_state.domain_context = new_context.strip()
                    st.session_state.domain_context_editing = False
                    
                    # Domain context is set during video upload, no need to update
                    st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.domain_context_editing = False
                st.rerun()


def show_chat_interface():
    """Show enhanced chat interface."""
    # Header with New Chat Session button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 💬 Chat with Video")
    with col2:
        if st.button("🔄 New Chat Session", use_container_width=True, help="Start a completely new session (resets everything and returns to start)"):
            # Reset entire application state for a fresh start
            st.session_state.chat_history = []
            st.session_state.uploaded_videos = {}
            st.session_state.active_video_id = None
            st.session_state.domain_context = None
            st.session_state.domain_set = False
            st.session_state.domain_context_editing = False
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    
    # Check if video is selected and processed
    if not st.session_state.active_video_id:
        st.warning("📹 No video available. Please upload a video first.")
        return
    
    video_info = st.session_state.uploaded_videos.get(st.session_state.active_video_id)
    if not video_info:
        st.warning("📹 No video available. Please upload a video first.")
        return
    
    # Show processing message below title if video is still processing
    if video_info.get("status") != "completed":
        st.info("⏳ **Video is still processing...** Please wait for processing to complete before chatting.")
        
        # Check if any videos are processing (same logic as sidebar)
        has_processing = any(v.get("status") == "processing" for v in st.session_state.uploaded_videos.values())
        
        # Manual refresh button - make it more prominent when processing (same as sidebar)
        if has_processing:
            if st.button("🔄 Refresh Status", key="refresh_status_chat", help="Click to check if video processing is complete", type="primary", use_container_width=True):
                st.rerun()
            st.caption("💡 Click Refresh Status to check processing progress")
        else:
            if st.button("🔄 Refresh Status", key="refresh_status_chat", help="Manually refresh video processing status", use_container_width=True):
                st.rerun()
        
        return
    
    # Determine required indexes based on whether domain context was provided
    has_domain_context = bool(st.session_state.get("domain_context") and 
                              st.session_state.domain_context != "General video analysis")
    required_indexes = get_required_indexes(has_domain_context)
    
    # Check if video is completed (allow partial completion)
    indexes = video_info.get("indexes", [])
    index_names = [idx.value if hasattr(idx, 'value') else str(idx).upper() for idx in indexes]
    has_required_indexes = all(idx in index_names for idx in required_indexes)

    # Show warning if some indexes are missing but allow use of available indexes
    if not has_required_indexes:
        missing_indexes = [idx for idx in required_indexes if idx not in index_names]
        st.warning(f"⚠️ Some indexes failed to create: {', '.join(missing_indexes)}. You can still use the video with available indexes.")
        st.info("💡 The video is usable but may have reduced search capabilities.")
    
    
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
        st.info("💬 Ask questions about the video content")
    
    # Chat input
    user_query = st.chat_input("Ask a question about the video...")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Call chat API
        chat_error = None
        with st.spinner("🤔 Thinking..."):
            try:
                current_url = get_api_base_url()
                response = requests.post(
                    f"{current_url}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "video_id": st.session_state.active_video_id,
                        "query": user_query,
                        "domain_context": st.session_state.domain_context,
                    },
                    timeout=120  # Chat can take longer
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": data.get("answer", "No answer provided"),
                        "citations": data.get("citations", []),
                    })
                else:
                    error_msg = f"Backend returned error {response.status_code}"
                    if response.text:
                        error_msg += f": {response.text[:200]}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {error_msg}",
                    })
            except requests.exceptions.ConnectionError as e:
                current_url = get_api_base_url()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Cannot connect to backend at `{current_url}`. Please check your connection and ensure the backend is running.",
                })
                chat_error = "connection"
            except requests.exceptions.Timeout as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "❌ Request timed out. The backend may be slow or unresponsive. Please try again.",
                    })
                chat_error = "timeout"
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                })
                chat_error = "error"
        
        # Show retry button if there was an error
        if chat_error:
            if chat_error == "connection":
                st.error("❌ Connection failed. Click Retry to try again.")
            elif chat_error == "timeout":
                st.error("❌ Request timed out. Click Retry to try again.")
            else:
                st.error("❌ An error occurred. Click Retry to try again.")
            
            if st.button("🔄 Retry Chat", key="retry_chat", use_container_width=True, type="primary"):
                st.rerun()
        else:
            st.rerun()


def main():
    """Main application."""
    initialize_session_state()
    
    # Show connection status in sidebar
    show_connection_status()
    
    # Show domain context dialog if not set
    if not st.session_state.domain_set:
        show_domain_context_dialog()
        return
    
    # Main interface
    st.markdown('<div class="main-header">🎬 QuadRAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Four-Index Multimodal RAG System for Video Understanding</div>', unsafe_allow_html=True)
    
    # Sidebar
    show_video_library()
    show_system_info()  # Move to sidebar
    
    # Auto-select video if there's only one video and none is currently selected
    if not st.session_state.active_video_id and len(st.session_state.uploaded_videos) == 1:
        # Automatically select the only video
        video_id = list(st.session_state.uploaded_videos.keys())[0]
        st.session_state.active_video_id = video_id
    
    # Domain Context (at the top, above upload)
    show_domain_context_panel()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content - Upload Video (prominent)
    uploaded_file = upload_video_section()
    
    # Process upload if file selected
    if uploaded_file is not None:
        # Validate file extension
        file_extension = uploaded_file.name.lower().split('.')[-1] if '.' in uploaded_file.name else ''
        if file_extension != 'mp4':
            st.error(f"❌ Invalid file format. Only .mp4 files are supported. You uploaded: .{file_extension}")
            st.info("Please upload a file with .mp4 extension")
            return
        
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📹 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        
        if st.button("🚀 Upload and Process", type="primary", use_container_width=True):
            # Check connection first
            current_url = get_api_base_url()
            is_connected, message = check_api_connection()
            if not is_connected:
                st.error(f"❌ Cannot connect to backend: {message}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔄 Retry Connection", use_container_width=True, type="primary"):
                        st.rerun()
                with col2:
                    if st.button("🔧 Check Connection Details", use_container_width=True):
                        st.info(f"**Backend URL:** `{current_url}`\n\n**Status:** {message}\n\n**Troubleshooting:**\n1. Verify Railway backend is running\n2. Check your internet connection\n3. Backend URL is correct")
                return
            
            with st.spinner("📤 Uploading video..."):
                try:
                    # Upload video
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{current_url}/upload-video", files=files, timeout=60)

                    if response.status_code == 200:
                        data = response.json()
                        video_id = data["video_id"]

                        st.success(f"✅ Video uploaded successfully!")
                        st.info("🎬 Video is now being processed in the background. This may take several minutes for large videos.")

                        # Processing starts automatically in background - just update session state
                        st.session_state.uploaded_videos[video_id] = {
                            "filename": uploaded_file.name,
                            "upload_time": datetime.now(),
                            "status": "processing",  # Background processing already started
                            "indexes": [],
                            "domain_context": st.session_state.get("domain_context"),
                            "session_id": st.session_state.get("session_id"),
                        }
                        # Automatically select the newly uploaded video
                        st.session_state.active_video_id = video_id

                        # Show progress message
                        with st.spinner("⏳ Background processing started..."):
                            # Give the backend a moment to start processing
                            import time
                            time.sleep(2)

                        st.success("🚀 Processing initiated! Check the video status in the sidebar.")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to upload video: Status {response.status_code}")
                        if response.text:
                            st.error(f"Error: {response.text}")
                        if st.button("🔄 Retry Upload", use_container_width=True, type="primary"):
                            st.rerun()
                except requests.exceptions.ConnectionError:
                    current_url = get_api_base_url()
                    st.error(f"❌ Cannot connect to backend at `{current_url}`")
                    st.info("**Please check:**\n1. Railway backend is running\n2. Your internet connection is active\n3. Backend URL is correct")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("🔄 Retry Connection", use_container_width=True, type="primary"):
                            st.rerun()
                    with col2:
                        if st.button("🔧 Connection Help", use_container_width=True):
                            with st.expander("Connection Troubleshooting", expanded=True):
                                st.markdown(f"""
                                **Backend URL:** `{current_url}`
                                
                                **Steps to fix:**
                                1. Check Railway dashboard to ensure backend is running
                                2. Verify your internet connection
                                3. Try refreshing the page
                                4. Check if backend URL is correct in environment variables
                                """)
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. The backend may be slow or unresponsive.")
                    if st.button("🔄 Retry Upload", use_container_width=True, type="primary"):
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    if st.button("🔄 Retry", use_container_width=True, type="primary"):
                        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    # Chat Interface
    show_chat_interface()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("QuadRAG v0.1.0")


if __name__ == "__main__":
    main()
