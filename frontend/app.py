"""QuadRAG — Streamlit frontend.

Redesign goals (vs. the pre-2026-04-22 revision):

* Linear flow: upload → status → chat. One main area at a time.
* Calm visual language. No rainbow gradients, generous whitespace, muted
  semantic colors, and Streamlit's native chat / status / expander widgets
  where they beat hand-rolled HTML.
* Honest status reporting. The backend's domain view is lazy-built on first
  chat with a new domain_context, so "DOMAIN index missing" is the normal
  pre-chat state and must NOT be surfaced as a failure.
* Forgiving connection handling. The old sidebar flipped to "Backend
  Disconnected" on the first slow probe (Railway cold-start, typically ~3 s
  over the edge proxy). Now we only alarm after two consecutive confirmed
  failures, keep the dot subtle, and never contradict the main area.

All env var overrides (QUADRAG_API_URL, QUADRAG_*_TIMEOUT_SEC,
QUADRAG_STATUS_POLL_INTERVAL_SEC) keep their Step-12 semantics.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BACKEND_URL = "http://localhost:8000"

# Env-var overrides — see CLAUDE.md §5
API_BASE_URL = os.getenv("QUADRAG_API_URL", DEFAULT_BACKEND_URL)
CHAT_TIMEOUT_SEC = int(os.getenv("QUADRAG_CHAT_TIMEOUT_SEC", "120"))
UPLOAD_TIMEOUT_SEC = int(os.getenv("QUADRAG_UPLOAD_TIMEOUT_SEC", "60"))
STATUS_POLL_TIMEOUT_SEC = int(os.getenv("QUADRAG_STATUS_POLL_TIMEOUT_SEC", "10"))
STATUS_POLL_INTERVAL_SEC = float(os.getenv("QUADRAG_STATUS_POLL_INTERVAL_SEC", "2.0"))
CONNECTION_CHECK_INTERVAL_SEC = 15.0

# Indexes the backend guarantees to build eagerly at upload time.
# DOMAIN is lazy-built on the first /chat with a domain_context, so it's
# intentionally NOT in this list — treating a missing DOMAIN at upload time
# as a failure is what the old UI got wrong.
EAGER_INDEXES = ["IMAGE", "AUDIO", "DESCRIPTION"]

st.set_page_config(
    page_title="QuadRAG · Video Understanding",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Minimal CSS — Streamlit's native theme handles most of it (see
# .streamlit/config.toml). Only small surgical tweaks live here.
# ---------------------------------------------------------------------------

_CSS = """
<style>
  /* Hide the default Streamlit top bar hamburger/menu when embedded. */
  [data-testid="stToolbar"] { visibility: hidden; height: 0; }

  /* Tighter main-area top padding — the default wastes ~80px. */
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

  /* Status dot in header. */
  .status-dot {
    display: inline-block;
    width: 0.55rem; height: 0.55rem;
    border-radius: 50%;
    margin-right: 0.4rem;
    vertical-align: middle;
  }
  .status-ok   { background: #16a34a; }
  .status-warn { background: #eab308; }
  .status-err  { background: #dc2626; }

  /* Index-pill badges. */
  .idx-pill {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    margin-right: 0.35rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
  }
  .idx-on   { background: #eef2ff; color: #4338ca; border-color: #c7d2fe; }
  .idx-lazy { background: #f3f4f6; color: #374151; border-color: #e5e7eb; }
  .idx-off  { background: #fef2f2; color: #991b1b; border-color: #fecaca; }

  /* Subtle citation block. */
  .citation-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #4f46e5;
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    line-height: 1.45;
  }
  .citation-meta { color: #64748b; font-size: 0.75rem; margin-bottom: 0.2rem; }
  .citation-meta .src { font-weight: 600; color: #4338ca; margin-right: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; }

  /* "Ungrounded" subtle pill above an answer. */
  .ungrounded-note {
    display: inline-block;
    font-size: 0.78rem;
    color: #92400e;
    background: #fffbeb;
    border: 1px solid #fde68a;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    margin-bottom: 0.6rem;
  }

  /* Footer hint. */
  .app-footer {
    color: #64748b;
    font-size: 0.78rem;
    text-align: center;
    padding-top: 2rem;
  }

  /* Hero (empty state) */
  .hero-title { font-size: 1.75rem; font-weight: 600; margin: 0 0 0.4rem; letter-spacing: -0.02em; }
  .hero-sub   { color: #475569; font-size: 0.98rem; margin: 0 0 1.6rem; }
</style>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "domain_context": "",
        "active_video_id": None,
        "chat_history": [],
        "uploaded_videos": {},         # video_id → {filename, upload_time, status, indexes, index_errors, grounded_domain}
        "last_status_poll": {},        # video_id → unix ts
        "last_connection_check": 0.0,
        "connection_ok": True,
        "connection_msg": "",
        "connection_fail_streak": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# API client (tiny, typed, tolerant)
# ---------------------------------------------------------------------------

@dataclass
class ConnectionState:
    ok: bool
    label: str


def _api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def check_backend(force: bool = False) -> ConnectionState:
    """Probe /health. Only updates state if enough time has elapsed — avoids
    hammering the backend on every Streamlit rerun. Only alarms after two
    consecutive failures, so a single slow probe doesn't flip the UI red."""
    now = time.time()
    if not force and (now - st.session_state.last_connection_check) < CONNECTION_CHECK_INTERVAL_SEC:
        return ConnectionState(st.session_state.connection_ok, st.session_state.connection_msg)

    st.session_state.last_connection_check = now
    try:
        r = requests.get(_api_url("/health"), timeout=STATUS_POLL_TIMEOUT_SEC)
        if r.status_code == 200:
            st.session_state.connection_ok = True
            st.session_state.connection_msg = "Connected"
            st.session_state.connection_fail_streak = 0
        else:
            st.session_state.connection_fail_streak += 1
            if st.session_state.connection_fail_streak >= 2:
                st.session_state.connection_ok = False
                st.session_state.connection_msg = f"Backend returned {r.status_code}"
    except requests.exceptions.RequestException as e:
        st.session_state.connection_fail_streak += 1
        if st.session_state.connection_fail_streak >= 2:
            st.session_state.connection_ok = False
            # Shorten the exception message; the full urllib3 wall of text isn't useful.
            brief = type(e).__name__
            st.session_state.connection_msg = f"Connection issue ({brief})"

    return ConnectionState(st.session_state.connection_ok, st.session_state.connection_msg)


def upload_video(file) -> Optional[dict[str, Any]]:
    try:
        r = requests.post(
            _api_url("/upload-video"),
            files={"file": (file.name, file.getvalue())},
            timeout=UPLOAD_TIMEOUT_SEC,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"Upload failed ({r.status_code}): {r.text[:300]}")
    except requests.exceptions.Timeout:
        st.error(f"Upload timed out after {UPLOAD_TIMEOUT_SEC}s. Try a shorter video.")
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach backend at {API_BASE_URL}.")
    except Exception as e:
        st.error(f"Upload error: {e}")
    return None


def _should_refetch_status(video_id: str) -> bool:
    last = st.session_state.last_status_poll.get(video_id, 0.0)
    if time.time() - last >= STATUS_POLL_INTERVAL_SEC:
        st.session_state.last_status_poll[video_id] = time.time()
        return True
    return False


def refresh_video_status(video_id: str) -> dict[str, Any]:
    """Fetch /status if the debounce allows; otherwise return the cached view.
    Always returns a dict with keys: status, indexes, index_errors."""
    cached = st.session_state.uploaded_videos.get(video_id, {})
    if not _should_refetch_status(video_id):
        return {
            "status": cached.get("status", "unknown"),
            "indexes": cached.get("indexes", []),
            "index_errors": cached.get("index_errors", {}),
        }
    try:
        r = requests.get(_api_url(f"/video/{video_id}/status"), timeout=STATUS_POLL_TIMEOUT_SEC)
        if r.status_code == 200:
            data = r.json()
            fresh = {
                "status": data.get("status", "unknown"),
                "indexes": [str(i).upper().replace("INDEXTYPE.", "") for i in data.get("indexes_created", [])],
                "index_errors": data.get("index_errors", {}),
            }
            slot = st.session_state.uploaded_videos.setdefault(video_id, {})
            slot.update(fresh)
            return fresh
    except requests.exceptions.RequestException:
        # Silent; keep cached state. Connection health is checked elsewhere.
        pass
    return {
        "status": cached.get("status", "unknown"),
        "indexes": cached.get("indexes", []),
        "index_errors": cached.get("index_errors", {}),
    }


def send_chat(video_id: str, query: str, domain_context: Optional[str]) -> Optional[dict[str, Any]]:
    payload = {
        "session_id": st.session_state.session_id,
        "video_id": video_id,
        "query": query,
    }
    if domain_context:
        payload["domain_context"] = domain_context
    try:
        r = requests.post(_api_url("/chat"), json=payload, timeout=CHAT_TIMEOUT_SEC)
        if r.status_code == 200:
            return r.json()
        return {"_error": f"Backend returned {r.status_code}", "_detail": r.text[:400]}
    except requests.exceptions.Timeout:
        return {"_error": "timeout", "_detail": (
            "The first query with a new domain context can take 30–60 s while "
            "Pixeltable builds that view. Try the same query again — it'll hit "
            "the cache and return in a couple of seconds."
        )}
    except requests.exceptions.ConnectionError:
        return {"_error": "connection", "_detail": f"Can't reach backend at {API_BASE_URL}."}
    except Exception as e:
        return {"_error": "error", "_detail": str(e)}


def reprocess_video(video_id: str, domain_context: Optional[str]) -> bool:
    try:
        r = requests.post(
            _api_url("/reprocess-video"),
            json={
                "video_id": video_id,
                "session_id": st.session_state.session_id,
                "domain_context": domain_context or None,
            },
            timeout=30,
        )
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def fmt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def has_custom_domain() -> bool:
    ctx = (st.session_state.domain_context or "").strip()
    return bool(ctx) and ctx.lower() != "general video analysis"


def effective_domain_context() -> Optional[str]:
    return st.session_state.domain_context.strip() or None if has_custom_domain() else None


def video_is_ready(indexes: list[str]) -> bool:
    """Ready once the three eager indexes exist. DOMAIN is lazy — never required."""
    return all(idx in indexes for idx in EAGER_INDEXES)


# ---------------------------------------------------------------------------
# UI: header
# ---------------------------------------------------------------------------

def render_header() -> None:
    conn = check_backend()
    dot_class = "status-ok" if conn.ok else "status-err"
    left, right = st.columns([7, 3])
    with left:
        st.markdown("## 🎬 QuadRAG")
        st.caption("Multimodal video understanding · audio · visual · semantic")
    with right:
        st.markdown(
            f"<div style='text-align:right;padding-top:1.1rem;'>"
            f"<span class='status-dot {dot_class}'></span>"
            f"<span style='color:#475569;font-size:0.85rem;'>{conn.label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()


# ---------------------------------------------------------------------------
# UI: sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.markdown("### Videos")

    videos = st.session_state.uploaded_videos
    if not videos:
        st.caption("No videos yet. Upload one on the right.")
    else:
        for video_id, info in list(videos.items()):
            fresh = refresh_video_status(video_id)
            info.update(fresh)
            _sidebar_video_entry(video_id, info)

        st.divider()
        if st.button("🧹 Clear session", use_container_width=True,
                     help="Forget uploaded videos + chat in this browser. Backend data stays."):
            st.session_state.uploaded_videos = {}
            st.session_state.active_video_id = None
            st.session_state.chat_history = []
            st.session_state.last_status_poll = {}
            st.rerun()

    st.divider()
    _sidebar_domain_panel()
    _sidebar_footer()


def _sidebar_video_entry(video_id: str, info: dict[str, Any]) -> None:
    status = info.get("status", "unknown")
    indexes = info.get("indexes", [])
    is_active = video_id == st.session_state.active_video_id
    name = info.get("filename", video_id[:8])

    ready = video_is_ready(indexes) or status == "completed"

    if status == "failed":
        status_line = "❌ Failed"
    elif ready:
        status_line = f"✅ Ready · {len([i for i in indexes if i in EAGER_INDEXES])}/3 indexes"
    elif status == "processing":
        status_line = "⏳ Processing…"
    else:
        status_line = f"· {status}"

    with st.container(border=True):
        st.markdown(
            f"**{'⭐ ' if is_active else ''}{_truncate(name, 34)}**  \n"
            f"<span style='color:#64748b;font-size:0.82rem'>{status_line}</span>",
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Open", key=f"open_{video_id}", use_container_width=True, disabled=is_active):
                st.session_state.active_video_id = video_id
                st.rerun()
        with cols[1]:
            if status == "processing":
                if st.button("↻ Refresh", key=f"refresh_{video_id}", use_container_width=True):
                    st.session_state.last_status_poll.pop(video_id, None)
                    st.rerun()
            else:
                if st.button("Reprocess", key=f"rep_{video_id}", use_container_width=True):
                    if reprocess_video(video_id, effective_domain_context()):
                        st.session_state.uploaded_videos[video_id]["status"] = "processing"
                        st.rerun()
                    else:
                        st.error("Reprocess failed.")


def _sidebar_domain_panel() -> None:
    st.markdown("### Domain context")
    if has_custom_domain():
        st.markdown(
            f"<div style='background:#eef2ff;border:1px solid #c7d2fe;padding:0.55rem 0.8rem;"
            f"border-radius:8px;color:#3730a3;font-size:0.85rem;line-height:1.45;'>"
            f"{st.session_state.domain_context}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Queries use this to build a dedicated domain index on first ask.")
    else:
        st.caption("Unset — chat uses the general-purpose indexes.")

    with st.expander("✏️ Edit", expanded=False):
        new_val = st.text_input(
            "Context",
            value=st.session_state.domain_context,
            placeholder="e.g. “luxury travel marketing”, “cooking techniques”",
            key="domain_input",
            label_visibility="collapsed",
        )
        cols = st.columns(2)
        with cols[0]:
            if st.button("Save", use_container_width=True):
                st.session_state.domain_context = new_val.strip()
                st.rerun()
        with cols[1]:
            if st.button("Clear", use_container_width=True, disabled=not st.session_state.domain_context):
                st.session_state.domain_context = ""
                st.rerun()


def _sidebar_footer() -> None:
    st.markdown("### Session")
    st.caption(f"ID: `{st.session_state.session_id[:8]}`")
    st.caption(f"Backend: `{API_BASE_URL}`")


# ---------------------------------------------------------------------------
# UI: empty state (no active video)
# ---------------------------------------------------------------------------

def render_empty_state() -> None:
    st.markdown("<p class='hero-title'>Ask questions about any video.</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>"
        "Upload an MP4 and QuadRAG indexes it four ways — visual frames, spoken audio, "
        "AI-generated descriptions, and an optional domain-specific lens. Then chat with it."
        "</p>",
        unsafe_allow_html=True,
    )
    _upload_widget()


def _upload_widget() -> None:
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4"],
            help="MP4 only, up to 500 MB, ≤ 2 hours.",
            key="uploader",
        )
        cols = st.columns([3, 2])
        with cols[0]:
            if uploaded_file:
                st.caption(f"**{uploaded_file.name}** · {uploaded_file.size / 1e6:.1f} MB")
        with cols[1]:
            if uploaded_file and st.button("Upload & index", type="primary", use_container_width=True):
                with st.spinner("Uploading…"):
                    resp = upload_video(uploaded_file)
                if resp:
                    vid = resp["video_id"]
                    st.session_state.uploaded_videos[vid] = {
                        "filename": uploaded_file.name,
                        "upload_time": datetime.now().isoformat(),
                        "status": "processing",
                        "indexes": [],
                        "index_errors": {},
                    }
                    st.session_state.active_video_id = vid
                    st.success("Upload complete — indexing in background.")
                    st.rerun()


# ---------------------------------------------------------------------------
# UI: active video + chat
# ---------------------------------------------------------------------------

def render_active_video() -> None:
    video_id = st.session_state.active_video_id
    info = st.session_state.uploaded_videos.get(video_id)
    if not info:
        st.session_state.active_video_id = None
        st.rerun()
        return

    fresh = refresh_video_status(video_id)
    info.update(fresh)

    status = info.get("status", "unknown")
    indexes = info.get("indexes", [])
    index_errors = info.get("index_errors", {})
    ready = video_is_ready(indexes) or status == "completed"

    # --- header strip ---
    st.markdown(f"### {info.get('filename', video_id[:8])}")
    _render_index_pills(indexes, index_errors)
    st.caption(f"`{video_id}`")

    if not ready and status == "processing":
        with st.status("Processing video…", expanded=True) as s:
            st.write("Transcoding + frame sampling + CLIP embeddings")
            st.write("Whisper transcription + text-embedding index")
            st.write("Frame descriptions via vision LLM + embedding index")
            st.caption("A 20-second clip typically finishes in ~30 s. Longer videos scale roughly linearly.")
            if st.button("↻ Check now"):
                st.session_state.last_status_poll.pop(video_id, None)
                st.rerun()
        return

    if status == "failed":
        st.error(f"Processing failed: {info.get('error') or 'unknown error'}")
        if st.button("Try reprocess"):
            if reprocess_video(video_id, effective_domain_context()):
                st.session_state.uploaded_videos[video_id]["status"] = "processing"
                st.rerun()
        return

    # Real per-eager-index errors → show, but don't cry about DOMAIN missing.
    real_errors = {k: v for k, v in index_errors.items() if k.upper() in EAGER_INDEXES}
    if real_errors:
        with st.expander(f"⚠️ {len(real_errors)} index error(s)", expanded=False):
            for k, v in real_errors.items():
                st.error(f"**{k}**: {v}")
            if st.button("Reprocess failed indexes"):
                if reprocess_video(video_id, effective_domain_context()):
                    st.session_state.uploaded_videos[video_id]["status"] = "processing"
                    st.rerun()

    st.divider()
    _render_chat_panel(video_id)


def _render_index_pills(indexes: list[str], index_errors: dict[str, str]) -> None:
    """Render index badges. DOMAIN is always 'lazy' unless already built.

    Previously: if DOMAIN wasn't in `indexes`, the UI warned "DOMAIN failed."
    Post-Step-8 that's wrong — DOMAIN builds on first chat with a
    domain_context, so "not present at upload" is expected, not failed.
    """
    pills_html: list[str] = []
    for name in EAGER_INDEXES:
        if name in indexes:
            css = "idx-on"
        elif name in index_errors:
            css = "idx-off"
        else:
            css = "idx-lazy"
        pills_html.append(f"<span class='idx-pill {css}'>{name}</span>")

    # Domain — render state separately and honestly.
    if "DOMAIN" in indexes:
        pills_html.append("<span class='idx-pill idx-on'>DOMAIN</span>")
    elif has_custom_domain():
        pills_html.append("<span class='idx-pill idx-lazy'>DOMAIN · on first query</span>")
    else:
        pills_html.append("<span class='idx-pill idx-lazy'>DOMAIN · off</span>")

    st.markdown(" ".join(pills_html), unsafe_allow_html=True)


def _render_chat_panel(video_id: str) -> None:
    """Chat messages + input. Uses Streamlit's native st.chat_* widgets."""
    cols = st.columns([6, 1])
    with cols[0]:
        st.markdown("#### Chat")
    with cols[1]:
        if st.button("Clear", use_container_width=True, help="Clear chat history (local)"):
            st.session_state.chat_history = []
            st.rerun()

    for msg in st.session_state.chat_history:
        role = msg["role"]
        with st.chat_message(role):
            if role == "assistant" and msg.get("grounded") is False:
                st.markdown(
                    "<div class='ungrounded-note'>Ungrounded · the model didn't cite specific timestamps</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            cites = msg.get("citations") or []
            if cites:
                with st.expander(f"Citations ({len(cites)})", expanded=False):
                    for c in cites:
                        _render_citation(c)

    placeholder = "Ask about the video…"
    if has_custom_domain():
        placeholder = f"Ask about the video (lens: {st.session_state.domain_context})…"

    query = st.chat_input(placeholder)
    if not query:
        return

    st.session_state.chat_history.append({"role": "user", "content": query})
    domain = effective_domain_context()
    spinner_msg = "Thinking…"
    if domain and not _has_domain_been_built(video_id, domain):
        spinner_msg = "Building the domain view (30–60 s on first query)…"

    with st.spinner(spinner_msg):
        resp = send_chat(video_id, query, domain)

    if not resp:
        return

    if "_error" in resp:
        err = resp["_error"]
        detail = resp.get("_detail", "")
        if err == "timeout" and domain:
            _note_domain_build_attempt(video_id, domain)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"⚠️ **{err}** — {detail}",
            "grounded": None,
        })
    else:
        if domain:
            _note_domain_build_attempt(video_id, domain, succeeded=True)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": resp.get("answer", "(no answer)"),
            "citations": resp.get("citations", []),
            "grounded": resp.get("grounded"),
        })
    st.rerun()


def _render_citation(c: dict[str, Any]) -> None:
    src = str(c.get("source", "?")).upper().replace("INDEXTYPE.", "")
    ts = fmt_timestamp(float(c.get("timestamp", 0)))
    sim = float(c.get("similarity", 0.0))
    content = str(c.get("content", "")).strip().replace("\n", " ")
    if len(content) > 320:
        content = content[:317] + "…"
    st.markdown(
        f"<div class='citation-card'>"
        f"<div class='citation-meta'><span class='src'>{src}</span> "
        f"@ {ts} · similarity {sim:.2f}</div>"
        f"{content}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Lightweight tracking of which (video, domain) pairs we've already tried.
# Not authoritative — the backend is the source of truth — just so we can
# pick better spinner text on the first domain-aware query.
# ---------------------------------------------------------------------------

def _has_domain_been_built(video_id: str, domain: str) -> bool:
    seen = st.session_state.uploaded_videos.get(video_id, {}).get("domain_attempts", set())
    return domain in seen


def _note_domain_build_attempt(video_id: str, domain: str, succeeded: bool = False) -> None:
    slot = st.session_state.uploaded_videos.setdefault(video_id, {})
    seen = slot.setdefault("domain_attempts", set())
    if succeeded or domain in seen:
        seen.add(domain)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    render_header()

    with st.sidebar:
        render_sidebar()

    if not st.session_state.active_video_id:
        render_empty_state()
    else:
        render_active_video()

    # Bottom: tiny upload affordance when a video is already active.
    if st.session_state.active_video_id:
        with st.expander("➕ Upload another video"):
            _upload_widget()

    st.markdown("<div class='app-footer'>QuadRAG · four-index multimodal RAG for video</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
else:
    main()
