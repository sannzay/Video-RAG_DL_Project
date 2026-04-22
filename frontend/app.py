"""QuadRAG — Streamlit frontend.

Design notes after the second UX pass:

* Upload lives at the **top of the sidebar** like a "New chat" button, not in
  the main content area. Clicking it opens a modal that asks for a domain
  lens *first*, then the MP4 — enforcing the correct chronology (pick the
  lens → upload → index → chat).
* Domain context is **per-video**, not a session-global. Each video remembers
  the lens it was indexed with; chat queries use that lens automatically.
* Chat history is **keyed by video_id** so switching videos shows the right
  conversation, not a shared jumble.
* Processing status **auto-polls** via ``@st.fragment(run_every=5)``. No more
  "click Refresh". The main area flips to the chat view as soon as the
  eager indexes finish.
* ``[M:SS]`` citations in the answer text are **escaped before
  markdown rendering** so back-to-back references don't get swallowed as
  reference-link syntax. Grounding detection runs on the raw text, so the
  grounded flag stays accurate.
* ``st.chat_input`` gets a prominent border + shadow so it doesn't disappear
  at the bottom of the page.

Connection handling, environment var overrides, and the grounding helpers
are unchanged from the first redesign.
"""

from __future__ import annotations

import os
import re
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
API_BASE_URL = os.getenv("QUADRAG_API_URL", DEFAULT_BACKEND_URL)
CHAT_TIMEOUT_SEC = int(os.getenv("QUADRAG_CHAT_TIMEOUT_SEC", "120"))
UPLOAD_TIMEOUT_SEC = int(os.getenv("QUADRAG_UPLOAD_TIMEOUT_SEC", "60"))
STATUS_POLL_TIMEOUT_SEC = int(os.getenv("QUADRAG_STATUS_POLL_TIMEOUT_SEC", "10"))
STATUS_POLL_INTERVAL_SEC = float(os.getenv("QUADRAG_STATUS_POLL_INTERVAL_SEC", "2.0"))
CONNECTION_CHECK_INTERVAL_SEC = 15.0
AUTO_POLL_EVERY_SEC = 5.0

# Indexes the backend guarantees to build eagerly at upload time. DOMAIN is
# lazy — built on first chat with a domain_context — so it's intentionally
# absent from this list. The old UI treating "DOMAIN not present at upload"
# as a failure is what produced the "DOMAIN failed" false alarm.
EAGER_INDEXES = ["IMAGE", "AUDIO", "DESCRIPTION"]

st.set_page_config(
    page_title="QuadRAG · Video Understanding",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Minimal CSS — baseline theme is in .streamlit/config.toml.
# ---------------------------------------------------------------------------

_CSS = """
<style>
  /* Keep Streamlit's top toolbar visible — it holds the sidebar expand/collapse
     button. Hiding stToolbar globally (as we did before) also hid the "open
     sidebar" chevron when the user collapsed it, leaving no way to reopen. */
  .block-container { padding-top: 1.5rem; padding-bottom: 6rem; max-width: 1100px; }

  /* Belt-and-suspenders: explicitly ensure the sidebar collapse/expand button
     is visible on newer Streamlit versions that nest it differently. */
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"] {
    visibility: visible !important;
    display: block !important;
  }

  /* Sidebar new-video button styling — makes it feel like ChatGPT's "New chat". */
  [data-testid="stSidebar"] .stButton > button[kind="primary"] {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 0.8rem;
  }

  /* Status dot in header. */
  .status-dot { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%; margin-right: 0.4rem; vertical-align: middle; }
  .status-ok   { background: #16a34a; }
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

  /* Citation cards. */
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

  /* Timestamp chips inside answers. */
  .ts-chip {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
    padding: 0.05rem 0.4rem;
    margin: 0 0.1rem;
    border-radius: 5px;
    font-size: 0.82em;
    font-family: ui-monospace, Menlo, monospace;
  }

  /* Make st.chat_input pop — was invisible at the bottom. */
  [data-testid="stChatInput"],
  [data-testid="stChatInputContainer"] {
    border: 2px solid #4f46e5 !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.15) !important;
    background: #ffffff !important;
  }
  [data-testid="stChatInput"] textarea {
    font-size: 1rem !important;
    padding: 0.85rem 1rem !important;
  }
  [data-testid="stChatInput"]:focus-within {
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25) !important;
  }

  /* Sticky-ish bottom fade so the chat input doesn't blur into content above. */
  [data-testid="stBottomBlockContainer"] {
    background: linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.95) 18%, #ffffff 40%);
    padding-top: 1rem;
  }

  .hero-title { font-size: 1.75rem; font-weight: 600; margin: 0 0 0.4rem; letter-spacing: -0.02em; }
  .hero-sub   { color: #475569; font-size: 0.98rem; margin: 0 0 1.6rem; }

  .app-footer { color: #64748b; font-size: 0.78rem; text-align: center; padding-top: 2rem; }
</style>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _empty_flow() -> dict[str, Any]:
    """Shape for the in-chat new-video wizard state machine.

    Lives in ``st.session_state.new_video_flow``. ``conversation`` is a list of
    ``{role, content}`` dicts that migrates into the video's chat history once
    indexing completes.
    """
    return {
        "active": False,
        "step": None,  # "await_file" | "await_domain" | "await_confirm" | "uploading" | "processing" | "failed"
        "file_name": None,
        "file_bytes": None,
        "file_size": 0,
        "domain": None,  # str or None; "" means explicitly skipped
        "video_id": None,
        "error": None,
        "conversation": [],
    }


def _init_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "active_video_id": None,
        "chat_history": {},            # video_id → list[message]
        "uploaded_videos": {},         # video_id → {filename, status, indexes, index_errors, domain_context, ...}
        "last_status_poll": {},        # video_id → unix ts
        "last_connection_check": 0.0,
        "connection_ok": True,
        "connection_msg": "",
        "connection_fail_streak": 0,
        "new_video_flow": _empty_flow(),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

@dataclass
class ConnectionState:
    ok: bool
    label: str


def _api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def check_backend(force: bool = False) -> ConnectionState:
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
            st.session_state.connection_msg = f"Connection issue ({type(e).__name__})"
    return ConnectionState(st.session_state.connection_ok, st.session_state.connection_msg)


def upload_video(filename: str, data_bytes: bytes, domain_context: Optional[str]) -> Optional[dict[str, Any]]:
    """POST /upload-video with raw bytes + filename.

    Takes raw bytes (not a Streamlit UploadedFile) so the wizard can keep the
    file in ``session_state`` across reruns — the UploadedFile handle is only
    live during the rerun where the uploader widget produced it.
    """
    data = {}
    if domain_context:
        data["domain_context"] = domain_context
        data["session_id"] = st.session_state.session_id
    try:
        r = requests.post(
            _api_url("/upload-video"),
            files={"file": (filename, data_bytes)},
            data=data,
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


def refresh_video_status(video_id: str, force: bool = False) -> dict[str, Any]:
    cached = st.session_state.uploaded_videos.get(video_id, {})
    if not force and not _should_refetch_status(video_id):
        return {
            "status": cached.get("status", "unknown"),
            "indexes": cached.get("indexes", []),
            "index_errors": cached.get("index_errors", {}),
        }
    if force:
        st.session_state.last_status_poll[video_id] = time.time()
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
        pass
    return {
        "status": cached.get("status", "unknown"),
        "indexes": cached.get("indexes", []),
        "index_errors": cached.get("index_errors", {}),
    }


def send_chat(video_id: str, query: str, domain_context: Optional[str]) -> Optional[dict[str, Any]]:
    payload = {"session_id": st.session_state.session_id, "video_id": video_id, "query": query}
    if domain_context:
        payload["domain_context"] = domain_context
    try:
        r = requests.post(_api_url("/chat"), json=payload, timeout=CHAT_TIMEOUT_SEC)
        if r.status_code == 200:
            return r.json()
        return {"_error": f"Backend returned {r.status_code}", "_detail": r.text[:400]}
    except requests.exceptions.Timeout:
        return {"_error": "timeout", "_detail": (
            "First query with a new domain context can take 30–60 s while "
            "Pixeltable builds that view. Try again — it'll hit the cache next time."
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
# Helpers
# ---------------------------------------------------------------------------

# Matches both bracketed timestamp shapes the LLM actually produces:
#   * ``[M:SS]`` / ``(M:SS)`` with optional ``.f`` fractional part
#   * ``[Ns]`` / ``[N.Ns]`` — seconds directly, because our prompt shows
#     retrieved chunks as ``At 127.9s:`` and the model often mirrors that.
# Kept in sync with ``backend/src/quadrag/generation/rag_generator.py`` so
# display highlighting and backend grounding agree on what counts as a
# citation.
_TIMESTAMP_RE = re.compile(
    r"[\[\(]"
    r"(?:"
    r"(\d+):(\d{2})(?:\.\d+)?"          # group 1,2 — M:SS form
    r"|"
    r"(\d+(?:\.\d+)?)s"                  # group 3   — Ns / N.Ns form
    r")"
    r"[\]\)]"
)


def fmt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def prepare_answer_for_markdown(raw: str) -> str:
    """Replace timestamp citations with styled chips before handing text to
    ``st.markdown``. Covers both ``[M:SS]`` and ``[Ns]`` forms, always rendering
    as ``M:SS`` for visual consistency in the chat bubble.

    Why: Streamlit's markdown parser interprets back-to-back bracket pairs
    (``[0:13][0:10]``) as reference-style link syntax and eats them — that
    used to produce the "Ungrounded" false positive where the raw answer
    contained valid citations but the rendered text looked truncated at ``[``.
    Substituting HTML spans sidesteps the bracket-parsing ambiguity and
    makes the timestamps visually stand out.
    """
    if not raw:
        return ""

    def _chip(match: re.Match) -> str:
        m_str, s_str, raw_seconds = match.group(1), match.group(2), match.group(3)
        if m_str and s_str:
            return f"<span class='ts-chip'>{m_str}:{s_str}</span>"
        if raw_seconds is not None:
            total = float(raw_seconds)
            m = int(total // 60)
            s = int(total % 60)
            return f"<span class='ts-chip'>{m}:{s:02d}</span>"
        return match.group(0)

    return _TIMESTAMP_RE.sub(_chip, raw)


def get_current_domain_context(video_id: str) -> Optional[str]:
    """Per-video domain lens. Empty string / 'General video analysis' → None."""
    info = st.session_state.uploaded_videos.get(video_id) or {}
    ctx = (info.get("domain_context") or "").strip()
    if not ctx or ctx.lower() == "general video analysis":
        return None
    return ctx


def video_is_ready(indexes: list[str]) -> bool:
    return all(idx in indexes for idx in EAGER_INDEXES)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# In-chat new-video wizard
#
# Flow (each step renders its widget inside a chat_message so it feels like the
# bot is walking the user through setup):
#
#   await_file    → bot: "Upload an MP4"  + [file_uploader]
#   await_domain  → bot: "Pick a lens"    + preset chips + custom input + skip
#   await_confirm → bot: "Ready to go?"   + [Start] [Change lens]
#   processing    → bot: "Indexing..."    + auto-polling progress (fragment)
#   (on completion) conversation migrates into chat_history[video_id] and the
#   wizard clears itself; the main dispatch falls through to the regular chat
#   view with the wizard transcript as the chat's opening exchange.
#
# The entire wizard disables ``st.chat_input`` so the user can't send real
# chat queries until the video is indexed.
# ---------------------------------------------------------------------------

DOMAIN_PRESETS: list[tuple[str, str]] = [
    ("😊 Emotions", "emotions, mood, and atmosphere of people and scenes"),
    ("🎬 Actions", "key actions and events happening in the scene"),
    ("📢 Marketing", "marketing angles, selling points, and branding"),
    ("📚 Educational", "educational concepts, key takeaways, and learning points"),
    ("🎭 Storytelling", "narrative structure, storytelling, and character development"),
    ("🏞️ Travel", "travel destinations, locations, and atmosphere"),
]


def start_new_video_flow() -> None:
    """Sidebar 'New video' entry point — resets the wizard and routes the main
    area to it on the next rerun."""
    st.session_state.new_video_flow = _empty_flow()
    flow = st.session_state.new_video_flow
    flow["active"] = True
    flow["step"] = "await_file"
    flow["conversation"] = [{
        "role": "assistant",
        "content": "Let's set up a new video! First, upload an MP4 file below.",
    }]
    st.session_state.active_video_id = None
    st.rerun()


def cancel_new_video_flow() -> None:
    st.session_state.new_video_flow = _empty_flow()
    st.rerun()


def _advance_to_domain(chosen_label: str, chosen_value: str) -> None:
    flow = st.session_state.new_video_flow
    flow["domain"] = chosen_value
    display = chosen_label if not chosen_value else f"{chosen_label} (“{chosen_value}”)"
    if chosen_label == "⏭️ Skip" and not chosen_value:
        display = "⏭️ *(skip — no domain lens)*"
    flow["conversation"].append({"role": "user", "content": display})
    flow["conversation"].append({
        "role": "assistant",
        "content": _confirm_message(flow),
    })
    flow["step"] = "await_confirm"
    st.rerun()


def _go_back_to_domain() -> None:
    flow = st.session_state.new_video_flow
    flow["domain"] = None
    # Trim the confirm + previous user-choice messages from the transcript.
    while flow["conversation"] and flow["conversation"][-1]["role"] != "assistant" or \
          (flow["conversation"] and flow["conversation"][-1].get("content", "").startswith("Ready")):
        popped = flow["conversation"].pop()
        if popped["role"] == "user":
            break
    flow["conversation"].append({
        "role": "assistant",
        "content": "No problem — pick a different domain lens, or skip.",
    })
    flow["step"] = "await_domain"
    st.rerun()


def _confirm_message(flow: dict[str, Any]) -> str:
    lens_desc = "*no domain lens*" if not flow["domain"] else f"lens **“{flow['domain']}”**"
    size_mb = (flow["file_size"] or 0) / 1e6
    return (
        f"Ready to go? I'll process **{flow['file_name']}** ({size_mb:.1f} MB) with "
        f"{lens_desc}. Hit **Start indexing** below when you're ready."
    )


def _start_upload_from_wizard() -> None:
    flow = st.session_state.new_video_flow
    if not flow["file_bytes"] or not flow["file_name"]:
        return
    flow["step"] = "uploading"
    resp = upload_video(flow["file_name"], flow["file_bytes"], flow["domain"] or None)
    if not resp:
        flow["error"] = "Upload failed"
        flow["step"] = "await_confirm"
        flow["conversation"].append({
            "role": "assistant",
            "content": "⚠️ Upload failed. You can retry with **Start indexing** or change the lens.",
        })
        st.rerun()
        return

    vid = resp["video_id"]
    flow["video_id"] = vid
    flow["step"] = "processing"
    st.session_state.uploaded_videos[vid] = {
        "filename": flow["file_name"],
        "upload_time": datetime.now().isoformat(),
        "status": "processing",
        "indexes": [],
        "index_errors": {},
        "domain_context": flow["domain"] or "",
    }
    st.session_state.chat_history.setdefault(vid, [])
    st.session_state.active_video_id = vid
    flow["conversation"].append({
        "role": "assistant",
        "content": "⏳ Uploaded. Now indexing — this usually takes 20–60 s. I'll tell you when it's ready.",
    })
    st.rerun()


def _finalize_wizard_if_ready() -> None:
    """If the wizard is in the processing step and the backend reports the
    eager indexes are built, migrate the wizard transcript into the video's
    chat_history and clear the wizard. Called at the top of the wizard
    renderer so each auto-poll rerun gets a chance to finalize."""
    flow = st.session_state.new_video_flow
    if flow["step"] != "processing" or not flow["video_id"]:
        return

    fresh = refresh_video_status(flow["video_id"], force=True)
    info = st.session_state.uploaded_videos.setdefault(flow["video_id"], {})
    info.update(fresh)
    status = fresh["status"]
    indexes = fresh["indexes"]

    ready = video_is_ready(indexes) or status == "completed"
    if status == "failed":
        flow["conversation"].append({
            "role": "assistant",
            "content": "❌ Indexing failed. You can try again from the sidebar (**Reprocess**).",
        })
    elif ready:
        flow["conversation"].append({
            "role": "assistant",
            "content": f"✅ Indexing complete — ask me anything about **{flow['file_name']}**!",
        })

    if status == "failed" or ready:
        # Migrate wizard transcript into this video's chat history so it
        # becomes the opening of the chat. Also stash the uploaded file bytes
        # so the regular chat view can offer the same mini-preview.
        st.session_state.chat_history[flow["video_id"]] = list(flow["conversation"])
        if flow.get("file_bytes"):
            info["file_bytes"] = flow["file_bytes"]
        st.session_state.new_video_flow = _empty_flow()
        st.toast("Video ready", icon="🎉")
        st.rerun()


def render_new_video_wizard() -> None:
    flow = st.session_state.new_video_flow

    # If we're in processing, check for completion on every render (this
    # function runs both on normal reruns and on the 5-second fragment tick).
    _finalize_wizard_if_ready()

    # Committed conversation so far
    for msg in flow["conversation"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    step = flow["step"]

    if step == "await_file":
        _render_step_await_file()
    elif step == "await_domain":
        _render_step_await_domain()
    elif step == "await_confirm":
        _render_step_await_confirm()
    elif step == "uploading":
        with st.chat_message("assistant"):
            st.markdown("Uploading…")
            st.progress(0.1, text="Sending file to backend")
    elif step == "processing":
        _render_step_processing()

    # Always-disabled chat input during the wizard. When the wizard finishes,
    # control flows back to render_active_video which renders its own enabled
    # chat_input.
    st.chat_input(
        "Finish the setup above to start chatting…",
        disabled=True,
        key=f"wiz_disabled_input_{flow.get('video_id', 'pre')}",
    )

    # Small "Cancel" escape hatch, placed below the chat_input via the
    # sidebar — see render_sidebar's "Cancel setup" button.


def _render_step_await_file() -> None:
    flow = st.session_state.new_video_flow
    with st.chat_message("assistant"):
        file = st.file_uploader(
            "Select MP4",
            type=["mp4"],
            key="wiz_file_uploader",
            label_visibility="collapsed",
            help="Up to 500 MB, ≤ 2 hours.",
        )
        if file is not None:
            # Capture bytes immediately — UploadedFile is only live this rerun.
            flow["file_name"] = file.name
            flow["file_size"] = file.size
            flow["file_bytes"] = file.getvalue()
            flow["conversation"].append({
                "role": "user",
                "content": f"📹 **{file.name}** · {file.size / 1e6:.1f} MB",
            })
            flow["conversation"].append({
                "role": "assistant",
                "content": (
                    "Great! Now pick a **domain lens** — it shapes how the model "
                    "captions each frame during indexing. Quick picks below, "
                    "or type your own / skip."
                ),
            })
            flow["step"] = "await_domain"
            st.rerun()


def _render_step_await_domain() -> None:
    with st.chat_message("assistant"):
        st.caption("Quick picks")
        # Render 6 presets in a 3×2 grid.
        for row_start in range(0, len(DOMAIN_PRESETS), 3):
            cols = st.columns(3)
            for offset in range(3):
                idx = row_start + offset
                if idx >= len(DOMAIN_PRESETS):
                    break
                label, value = DOMAIN_PRESETS[idx]
                if cols[offset].button(label, key=f"wiz_preset_{idx}", use_container_width=True):
                    _advance_to_domain(label, value)

        st.markdown("")
        st.caption("Or type your own lens")
        custom_cols = st.columns([3, 1])
        with custom_cols[0]:
            custom = st.text_input(
                "Custom lens",
                key="wiz_custom_domain",
                placeholder="e.g. cooking techniques, sports analysis",
                label_visibility="collapsed",
            )
        with custom_cols[1]:
            if st.button("Use", key="wiz_use_custom", use_container_width=True,
                         disabled=not (custom or "").strip()):
                _advance_to_domain("✏️ Custom", custom.strip())

        st.markdown("")
        skip_cols = st.columns([1, 3])
        with skip_cols[0]:
            if st.button("⏭️ Skip", key="wiz_skip", use_container_width=True):
                _advance_to_domain("⏭️ Skip", "")


def _render_video_preview(file_bytes: bytes) -> None:
    """Mini video preview, centered and width-constrained to ~60% of the
    bubble so it doesn't dominate the chat layout.

    ``st.video`` embeds an HTML5 ``<video>`` tag; starts paused + muted by
    default so it never autoplays sound at the user. The browser caches the
    bytes, so re-renders across reruns don't re-download — but we still take
    care to keep this OUT of the 5-second auto-polling fragment so the player
    doesn't reset while the progress bar ticks.
    """
    if not file_bytes:
        return
    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.video(file_bytes, muted=True)


def _render_step_await_confirm() -> None:
    flow = st.session_state.new_video_flow
    with st.chat_message("assistant"):
        if flow.get("file_bytes"):
            _render_video_preview(flow["file_bytes"])
            st.markdown("")  # spacer between preview and buttons
        cols = st.columns([2, 1])
        with cols[0]:
            if st.button("🚀 Start indexing", type="primary", use_container_width=True, key="wiz_start"):
                _start_upload_from_wizard()
        with cols[1]:
            if st.button("↩️ Change lens", use_container_width=True, key="wiz_change_lens"):
                _go_back_to_domain()


def _render_step_processing() -> None:
    """Outer (non-polling) shell for the processing step: renders the static
    parts (chat bubble wrapper + video preview + header caption) once per
    full rerun, then delegates the polling progress bar to a fragment.

    Keeping the video preview OUT of the fragment prevents the HTML5 video
    element from re-rendering every 5 seconds, which would reset the
    playhead and cause visible flicker.
    """
    flow = st.session_state.new_video_flow
    video_id = flow.get("video_id")
    if not video_id:
        return

    with st.chat_message("assistant"):
        if flow.get("file_bytes"):
            _render_video_preview(flow["file_bytes"])
            st.markdown("")  # spacer

        st.markdown(
            f"**Processing `{flow['file_name']}`…**  \n"
            f"<span style='color:#64748b;font-size:0.85rem;'>"
            f"Auto-checking every {int(AUTO_POLL_EVERY_SEC)} s. You can't send "
            f"messages until indexing completes.</span>",
            unsafe_allow_html=True,
        )
        _processing_poll_fragment(video_id)


@st.fragment(run_every=AUTO_POLL_EVERY_SEC)
def _processing_poll_fragment(video_id: str) -> None:
    """The only thing that ticks every 5 s. Finalization itself runs at the
    top of ``render_new_video_wizard`` via ``_finalize_wizard_if_ready``;
    this fragment just escalates to an app-level rerun when it sees the
    eager indexes are ready or the job failed."""
    fresh = refresh_video_status(video_id, force=True)
    info = st.session_state.uploaded_videos.setdefault(video_id, {})
    info.update(fresh)
    done = [i for i in fresh["indexes"] if i in EAGER_INDEXES]

    st.progress(
        len(done) / max(1, len(EAGER_INDEXES)),
        text=f"{len(done)}/{len(EAGER_INDEXES)} indexes built",
    )
    if fresh["index_errors"]:
        for k, v in fresh["index_errors"].items():
            if k.upper() in EAGER_INDEXES:
                st.warning(f"⚠️ **{k}**: {v[:200]}")

    ready = video_is_ready(fresh["indexes"]) or fresh["status"] == "completed"
    if ready or fresh["status"] == "failed":
        st.rerun(scope="app")


# ---------------------------------------------------------------------------
# Header
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
# Sidebar — "New video" button at top, then the video list
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    flow = st.session_state.new_video_flow
    wizard_active = flow.get("active", False)

    # "New video" button — styled like ChatGPT's "New chat". Disabled mid-wizard
    # so the user can't clobber an in-progress setup.
    if st.button(
        "➕ New video",
        use_container_width=True,
        type="primary",
        disabled=wizard_active,
        help=("Start a new video setup" if not wizard_active
              else "Finish or cancel the current setup first."),
    ):
        start_new_video_flow()

    if wizard_active:
        if st.button("✕ Cancel setup", use_container_width=True):
            cancel_new_video_flow()

    st.markdown("")  # spacer

    st.markdown("### Videos")
    videos = st.session_state.uploaded_videos
    if not videos:
        st.caption("No videos yet. Click “New video” to upload one.")
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
        st.session_state.chat_history = {}
        st.session_state.last_status_poll = {}
        st.rerun()

    st.markdown("### Session")
    st.caption(f"ID: `{st.session_state.session_id[:8]}`")
    st.caption(f"Backend: `{API_BASE_URL}`")


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
            f"**{'⭐ ' if is_active else ''}{_truncate(name, 32)}**  \n"
            f"<span style='color:#64748b;font-size:0.82rem'>{status_line}</span>",
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Open", key=f"open_{video_id}", use_container_width=True, disabled=is_active):
                st.session_state.active_video_id = video_id
                st.session_state.chat_history.setdefault(video_id, [])
                st.rerun()
        with cols[1]:
            if st.button("Reprocess", key=f"rep_{video_id}", use_container_width=True,
                         help="Retry the backend indexing pipeline on this video"):
                if reprocess_video(video_id, info.get("domain_context") or None):
                    st.session_state.uploaded_videos[video_id]["status"] = "processing"
                    st.rerun()
                else:
                    st.error("Reprocess failed.")


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def render_empty_state() -> None:
    st.markdown("<p class='hero-title'>Ask questions about any video.</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>"
        "QuadRAG indexes each MP4 four ways — visual frames, spoken audio, "
        "AI-generated descriptions, and an optional domain-specific lens — then "
        "lets you chat with it. I'll walk you through setup."
        "</p>",
        unsafe_allow_html=True,
    )
    if st.button("➕ New video", type="primary", use_container_width=False):
        start_new_video_flow()


# ---------------------------------------------------------------------------
# Active video — processing (fragment auto-polls) OR chat
# ---------------------------------------------------------------------------

def render_active_video() -> None:
    video_id = st.session_state.active_video_id
    info = st.session_state.uploaded_videos.get(video_id)
    if not info:
        st.session_state.active_video_id = None
        st.rerun()
        return

    _render_video_header(video_id, info)

    # Decide processing vs chat view. Re-check status once on entry (cheap).
    fresh = refresh_video_status(video_id, force=False)
    info.update(fresh)
    status = info.get("status", "unknown")
    indexes = info.get("indexes", [])
    ready = video_is_ready(indexes) or status == "completed"

    if status == "failed":
        st.error(f"Processing failed: {info.get('error') or 'unknown error'}")
        if st.button("Try reprocess"):
            if reprocess_video(video_id, info.get("domain_context") or None):
                st.session_state.uploaded_videos[video_id]["status"] = "processing"
                st.rerun()
        return

    if not ready:
        _render_processing_fragment(video_id)
        return

    # Surface real per-eager-index errors if any. DOMAIN missing ≠ failure.
    index_errors = info.get("index_errors", {})
    real_errors = {k: v for k, v in index_errors.items() if k.upper() in EAGER_INDEXES}
    if real_errors:
        with st.expander(f"⚠️ {len(real_errors)} index error(s)", expanded=False):
            for k, v in real_errors.items():
                st.error(f"**{k}**: {v}")
            if st.button("Reprocess failed indexes"):
                if reprocess_video(video_id, info.get("domain_context") or None):
                    st.session_state.uploaded_videos[video_id]["status"] = "processing"
                    st.rerun()

    st.divider()
    _render_chat(video_id)


def _render_video_header(video_id: str, info: dict[str, Any]) -> None:
    st.markdown(f"### {info.get('filename', video_id[:8])}")
    _render_index_pills(info.get("indexes", []), info.get("index_errors", {}),
                        has_domain=bool(info.get("domain_context")))
    meta_bits = [f"`{video_id}`"]
    if info.get("domain_context"):
        meta_bits.append(f"lens: *{info['domain_context']}*")
    st.caption(" · ".join(meta_bits))

    # Collapsible preview. Only shown when we actually have the file bytes in
    # session_state (i.e. this browser session was the one that uploaded). A
    # second browser opening the same backend wouldn't have the bytes and
    # simply won't see the expander — that's honest, not a bug.
    if info.get("file_bytes"):
        with st.expander("📺 Preview", expanded=False):
            _render_video_preview(info["file_bytes"])


def _render_index_pills(indexes: list[str], index_errors: dict[str, str], has_domain: bool) -> None:
    pills: list[str] = []
    for name in EAGER_INDEXES:
        if name in indexes:
            css = "idx-on"
        elif name in index_errors:
            css = "idx-off"
        else:
            css = "idx-lazy"
        pills.append(f"<span class='idx-pill {css}'>{name}</span>")
    if "DOMAIN" in indexes:
        pills.append("<span class='idx-pill idx-on'>DOMAIN</span>")
    elif has_domain:
        pills.append("<span class='idx-pill idx-lazy'>DOMAIN · on first query</span>")
    else:
        pills.append("<span class='idx-pill idx-lazy'>DOMAIN · off</span>")
    st.markdown(" ".join(pills), unsafe_allow_html=True)


@st.fragment(run_every=AUTO_POLL_EVERY_SEC)
def _render_processing_fragment(video_id: str) -> None:
    """Auto-polling processing view. Streamlit re-executes this fragment every
    ``AUTO_POLL_EVERY_SEC`` seconds without a full app rerun, so the user doesn't
    have to click "Refresh". When the eager indexes flip to ready we escalate to
    an app-level rerun so the outer layout switches to the chat view."""
    fresh = refresh_video_status(video_id, force=True)
    info = st.session_state.uploaded_videos.setdefault(video_id, {})
    info.update(fresh)
    status = fresh["status"]
    indexes = fresh["indexes"]

    if status == "failed":
        st.rerun(scope="app")

    if video_is_ready(indexes) or status == "completed":
        st.toast("✅ Processing complete", icon="🎉")
        st.rerun(scope="app")

    with st.status("Processing video…", expanded=True):
        st.write("Transcoding + frame sampling + CLIP embeddings")
        st.write("Whisper transcription + text embedding index")
        st.write("Frame descriptions via vision LLM + embedding index")
        st.caption(
            f"A 20-second clip typically finishes in ~30 s. Auto-checking every "
            f"{int(AUTO_POLL_EVERY_SEC)} seconds; no need to click anything."
        )
        done = [i for i in indexes if i in EAGER_INDEXES]
        st.progress(len(done) / max(1, len(EAGER_INDEXES)), text=f"{len(done)}/{len(EAGER_INDEXES)} indexes built")


def _render_chat(video_id: str) -> None:
    info = st.session_state.uploaded_videos.get(video_id, {})
    history = st.session_state.chat_history.setdefault(video_id, [])

    cols = st.columns([6, 1])
    with cols[0]:
        st.markdown("#### Chat")
    with cols[1]:
        if st.button("Clear chat", use_container_width=True, key=f"clear_chat_{video_id}",
                     help="Clear this video's chat history (local only)"):
            st.session_state.chat_history[video_id] = []
            st.rerun()

    for msg in history:
        role = msg["role"]
        with st.chat_message(role):
            if role == "assistant" and msg.get("grounded") is False:
                st.markdown(
                    "<div class='ungrounded-note'>Ungrounded · the model didn't cite specific timestamps</div>",
                    unsafe_allow_html=True,
                )
            # Render answer with timestamp-chip substitution so markdown doesn't
            # eat [0:13][0:10] as reference-link syntax.
            if role == "assistant":
                st.markdown(prepare_answer_for_markdown(msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
            cites = msg.get("citations") or []
            if cites:
                with st.expander(f"Citations ({len(cites)})", expanded=False):
                    for c in cites:
                        _render_citation(c)

    # Input
    domain = get_current_domain_context(video_id)
    placeholder = "Ask about the video…"
    if domain:
        placeholder = f"Ask about the video (lens: {domain})…"
    query = st.chat_input(placeholder)
    if not query:
        return

    history.append({"role": "user", "content": query})
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
        history.append({
            "role": "assistant",
            "content": f"⚠️ **{err}** — {detail}",
            "grounded": None,
        })
    else:
        if domain:
            _note_domain_build_attempt(video_id, domain, succeeded=True)
        history.append({
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
    # Escape any HTML chars in retrieved content so we don't inject user data.
    import html as _html
    safe = _html.escape(content)
    st.markdown(
        f"<div class='citation-card'>"
        f"<div class='citation-meta'><span class='src'>{src}</span> "
        f"@ {ts} · similarity {sim:.2f}</div>"
        f"{safe}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Small stateful helpers for "has the domain view been built this session"
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    render_header()

    with st.sidebar:
        render_sidebar()

    flow = st.session_state.new_video_flow
    if flow.get("active"):
        render_new_video_wizard()
    elif not st.session_state.active_video_id:
        render_empty_state()
    else:
        render_active_video()

    st.markdown("<div class='app-footer'>QuadRAG · four-index multimodal RAG for video</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
else:
    main()
