"""QuadRAG Testing UI - Compare different index combinations."""

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
    page_title="QuadRAG Testing - Index Comparison",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "QuadRAG Testing - Compare how different index combinations affect answers"
    }
)

# API endpoint - Fixed Railway backend URL
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

# Enhanced CSS with testing-specific styles
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        --danger-gradient: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
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

    /* Testing Cards - Different colors for each index combination */
    .test-card-audio {
        background: var(--success-gradient);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-lg);
        border: 2px solid rgba(79, 172, 254, 0.3);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .test-card-audio::before {
        content: "🎵";
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 2rem;
        opacity: 0.8;
    }

    .test-card-audio-desc {
        background: var(--warning-gradient);
        color: #333;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-lg);
        border: 2px solid rgba(252, 182, 159, 0.3);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .test-card-audio-desc::before {
        content: "🎵📝";
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 1.8rem;
        opacity: 0.8;
    }

    .test-card-complete {
        background: var(--secondary-gradient);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-lg);
        border: 2px solid rgba(245, 87, 108, 0.3);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .test-card-complete::before {
        content: "🎬";
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 2rem;
        opacity: 0.8;
    }

    /* Test Header */
    .test-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Answer Content */
    .answer-content {
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* Citations */
    .citation {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.75rem;
        padding: 0.75rem;
        background: rgba(255,255,255,0.1);
        border-left: 3px solid rgba(255,255,255,0.5);
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    .citation:hover {
        background: rgba(255,255,255,0.15);
        transform: translateX(4px);
    }

    /* Dark citations for audio-desc card */
    .citation-dark {
        color: rgba(0,0,0,0.8);
        background: rgba(255,255,255,0.8);
        border-left-color: rgba(0,0,0,0.3);
    }

    .citation-dark:hover {
        background: rgba(255,255,255,0.9);
    }

    /* Processing indicator */
    .processing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        background: rgba(255,255,255,0.2);
        color: white;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
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

    # Testing-specific state
    if "test_results" not in st.session_state:
        st.session_state.test_results = {}


def show_system_info():
    """Display system information and features."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 About Testing")
    st.sidebar.caption("Compare three index combinations:")
    st.sidebar.caption("• 🎵 Audio Only")
    st.sidebar.caption("• 🎵📝 Audio + Description")
    st.sidebar.caption("• 🎬 Complete QuadRAG")


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
        <p style="margin: 0; opacity: 0.9; font-size: 0.95rem;">Upload a <strong>.mp4</strong> video file to start testing</p>
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
    required_indexes = ["AUDIO", "IMAGE", "DESCRIPTION"]
    if has_domain_context:
        required_indexes.append("DOMAIN")

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
                required_indexes = ["AUDIO", "IMAGE", "DESCRIPTION"]
                if has_domain_context:
                    required_indexes.append("DOMAIN")

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
        required_indexes = ["AUDIO", "IMAGE", "DESCRIPTION"]
        if has_domain_context:
            required_indexes.append("DOMAIN")
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


def query_backend_with_indexes(video_id: str, query: str, domain_context: str, indexes_to_use: list):
    """Query the backend using specific indexes."""
    try:
        current_url = get_api_base_url()
        response = requests.post(
            f"{current_url}/chat",
            json={
                "session_id": st.session_state.session_id,
                "video_id": video_id,
                "query": query,
                "domain_context": domain_context,
                "indexes": indexes_to_use,  # We'll add this parameter
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Backend returned error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def show_testing_interface():
    """Show the testing interface with three different index combinations."""
    # Header with New Chat Session button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🧪 Index Comparison Testing")
    with col2:
        if st.button("🔄 New Session", use_container_width=True, help="Start a completely new session"):
            # Reset entire application state for a fresh start
            st.session_state.chat_history = []
            st.session_state.uploaded_videos = {}
            st.session_state.active_video_id = None
            st.session_state.domain_context = None
            st.session_state.domain_set = False
            st.session_state.domain_context_editing = False
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.test_results = {}
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
        st.info("⏳ **Video is still processing...** Please wait for processing to complete before testing.")

        # Check if any videos are processing (same logic as sidebar)
        has_processing = any(v.get("status") == "processing" for v in st.session_state.uploaded_videos.values())

        # Manual refresh button - make it more prominent when processing
        if has_processing:
            if st.button("🔄 Refresh Status", key="refresh_status_chat", help="Click to check if video processing is complete", type="primary", use_container_width=True):
                st.rerun()
            st.caption("💡 Click Refresh Status to check processing progress")
        else:
            if st.button("🔄 Refresh Status", key="refresh_status_chat", help="Manually refresh video processing status", use_container_width=True):
                st.rerun()

        return

    # Check if video has required indexes for testing
    indexes = video_info.get("indexes", [])
    # Convert indexes to lowercase strings for comparison
    index_names = []
    for idx in indexes:
        if isinstance(idx, str):
            # Already a string, convert to lowercase
            index_names.append(idx.lower())
        else:
            # Enum or other object, try to get string representation
            if hasattr(idx, 'value'):
                index_names.append(str(idx.value).lower())
            else:
                index_names.append(str(idx).lower())

    has_audio = "audio" in index_names
    has_description = "description" in index_names
    has_domain = "domain" in index_names

    if not has_audio:
        st.error("❌ This video doesn't have an Audio index. Cannot perform testing.")
        return

    # Query input
    st.markdown("#### Ask a Question")
    user_query = st.chat_input("Enter your question about the video...")

    if user_query:
        # Create a unique key for this query
        query_key = f"{user_query}_{int(time.time())}"

        # Initialize results for this query
        st.session_state.test_results[query_key] = {
            "query": user_query,
            "audio_only": {"status": "processing"},
            "audio_desc": {"status": "processing"},
            "complete": {"status": "processing"}
        }

        # Trigger rerun to show processing state
        st.rerun()

    # Display test results if any
    if st.session_state.test_results:
        for query_key, results in st.session_state.test_results.items():
            st.markdown("---")
            st.markdown(f"**Question:** {results['query']}")

            # Three columns for the three test cases
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="test-card-audio">', unsafe_allow_html=True)
                st.markdown('<div class="test-header">🎵 Audio Only</div>', unsafe_allow_html=True)

                if results["audio_only"]["status"] == "processing":
                    st.markdown('<div class="processing-indicator">⏳ Processing...</div>', unsafe_allow_html=True)
                elif "error" in results["audio_only"]:
                    st.error(f"❌ Error: {results['audio_only']['error']}")
                else:
                    answer = results["audio_only"].get("answer", "No answer provided")
                    st.markdown(f'<div class="answer-content">{answer}</div>', unsafe_allow_html=True)

                    # Citations
                    citations = results["audio_only"].get("citations", [])
                    if citations:
                        st.markdown("**Citations:**")
                        for citation in citations[:3]:  # Show first 3 citations
                            source = citation.get("source", "Unknown").upper()
                            timestamp = citation.get("timestamp", 0)
                            content = citation.get("content", "")[:100]
                            st.markdown(f'<div class="citation">[{source}] @{timestamp:.1f}s: {content}...</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="test-card-audio-desc">', unsafe_allow_html=True)
                st.markdown('<div class="test-header">🎵📝 Audio + Description</div>', unsafe_allow_html=True)

                if results["audio_desc"]["status"] == "processing":
                    st.markdown('<div class="processing-indicator" style="background: rgba(0,0,0,0.2); color: #333;">⏳ Processing...</div>', unsafe_allow_html=True)
                elif "error" in results["audio_desc"]:
                    st.error(f"❌ Error: {results['audio_desc']['error']}")
                else:
                    answer = results["audio_desc"].get("answer", "No answer provided")
                    st.markdown(f'<div class="answer-content">{answer}</div>', unsafe_allow_html=True)

                    # Citations
                    citations = results["audio_desc"].get("citations", [])
                    if citations:
                        st.markdown("**Citations:**")
                        for citation in citations[:3]:  # Show first 3 citations
                            source = citation.get("source", "Unknown").upper()
                            timestamp = citation.get("timestamp", 0)
                            content = citation.get("content", "")[:100]
                            st.markdown(f'<div class="citation citation-dark">[{source}] @{timestamp:.1f}s: {content}...</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="test-card-complete">', unsafe_allow_html=True)
                st.markdown('<div class="test-header">🎬 Complete QuadRAG</div>', unsafe_allow_html=True)

                if results["complete"]["status"] == "processing":
                    st.markdown('<div class="processing-indicator">⏳ Processing...</div>', unsafe_allow_html=True)
                elif "error" in results["complete"]:
                    st.error(f"❌ Error: {results['complete']['error']}")
                else:
                    answer = results["complete"].get("answer", "No answer provided")
                    st.markdown(f'<div class="answer-content">{answer}</div>', unsafe_allow_html=True)

                    # Citations
                    citations = results["complete"].get("citations", [])
                    if citations:
                        st.markdown("**Citations:**")
                        for citation in citations[:3]:  # Show first 3 citations
                            source = citation.get("source", "Unknown").upper()
                            timestamp = citation.get("timestamp", 0)
                            content = citation.get("content", "")[:100]
                            st.markdown(f'<div class="citation">[{source}] @{timestamp:.1f}s: {content}...</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        # Process any pending queries
        for query_key, results in st.session_state.test_results.items():
            if results["audio_only"]["status"] == "processing":
                # Query with only audio index
                result = query_backend_with_indexes(
                    st.session_state.active_video_id,
                    results["query"],
                    st.session_state.domain_context,
                    ["audio"]
                )
                if "error" in result:
                    results["audio_only"] = {"status": "error", "error": result["error"]}
                else:
                    results["audio_only"] = {"status": "completed", **result}

            if results["audio_desc"]["status"] == "processing":
                # Query with audio + description indexes
                indexes = ["audio", "description"]
                result = query_backend_with_indexes(
                    st.session_state.active_video_id,
                    results["query"],
                    st.session_state.domain_context,
                    indexes
                )
                if "error" in result:
                    results["audio_desc"] = {"status": "error", "error": result["error"]}
                else:
                    results["audio_desc"] = {"status": "completed", **result}

            if results["complete"]["status"] == "processing":
                # Query with all indexes
                indexes = ["audio", "description"]
                if has_domain:
                    indexes.append("domain")
                result = query_backend_with_indexes(
                    st.session_state.active_video_id,
                    results["query"],
                    st.session_state.domain_context,
                    indexes
                )
                if "error" in result:
                    results["complete"] = {"status": "error", "error": result["error"]}
                else:
                    results["complete"] = {"status": "completed", **result}

        # Check if any results changed and trigger rerun if needed
        all_completed = all(
            result["status"] in ["completed", "error"]
            for results in st.session_state.test_results.values()
            for result in results.values()
            if isinstance(result, dict) and "status" in result
        )
        if not all_completed:
            time.sleep(0.1)  # Small delay to prevent too frequent reruns
            st.rerun()

    else:
        # Show instructions
        st.info("💡 Enter a question above to compare how different index combinations affect the answers. Each combination will search different types of content from your video.")


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
    st.markdown('<div class="main-header">🧪 QuadRAG Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare how different index combinations affect video understanding answers</div>', unsafe_allow_html=True)

    # Sidebar
    show_video_library()
    show_system_info()

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

    # Testing Interface
    show_testing_interface()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("QuadRAG Testing v0.1.0")


if __name__ == "__main__":
    main()
