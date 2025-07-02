import streamlit as st
import os
import tempfile
import requests
import json
import time
from pathlib import Path
import uuid
import datetime as dt

# ── Local Imports ────────────────────────────────────────────────────────────────
from transcript_generator import WhisperTranscriber, setup_logging

# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lecture Transcript Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────────────────
HISTORY_FILE = "analysis_history.json"
MAX_TOKENS = 4000
MODEL_NAME = "sonar"                # cheapest Perplexity model
MAX_TRANSCRIPT_CHARS = 10_000       # conservative chunk for prompt


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           Utility: Progress Tracker                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class ProgressTracker:
    """Visual loader with dynamic ETA estimation."""
    def __init__(self) -> None:
        self.start_time: float | None = None
        self.progress_bar = None
        self.status_text = None
        self.total_steps: int = 0

    def initialize(self, total_steps: int) -> None:
        self.start_time = time.time()
        self.total_steps = max(total_steps, 1)
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def update(self, current_step: int, message: str) -> None:
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        pct = int(current_step / self.total_steps * 100)
        # simple ETA
        if current_step:
            eta = (elapsed / current_step) * (self.total_steps - current_step)
            eta_text = f"ETA: {eta:,.1f}s"
        else:
            eta_text = "Calculating ETA…"
        self.progress_bar.progress(min(pct, 100))
        self.status_text.info(f"🔄 {message}  ({current_step}/{self.total_steps}) • {eta_text}")

    def complete(self) -> None:
        if self.progress_bar:
            self.progress_bar.progress(100)
        if self.status_text:
            self.status_text.success("✅ Analysis Complete!")
        # brief pause so user sees completion message
        time.sleep(0.8)
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_text:
            self.status_text.empty()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       Utility: Persistent History                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def load_history() -> list:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_to_history(summary: str, resources: str | list, filename: str) -> None:
    history = load_history()
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "summary": summary,
        "resources": resources,
        "preview": (summary[:150] + "…") if len(summary) > 150 else summary,
    }
    history.append(entry)
    # limit to last 50 entries
    history = history[-50:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       Perplexity API Wrapper                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class PerplexityAPI:
    """Minimal wrapper around Perplexity chat completion endpoint with retries."""
    BASE_URL = "https://api.perplexity.ai"

    def __init__(self, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def chat_completion(
        self,
        messages: list[dict],
        model: str = MODEL_NAME,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = 3,
    ) -> str | None:
        url = f"{self.BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": False,
        }
        for attempt in range(max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=60)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                if resp.status_code == 429:  # rate-limited
                    wait = 2 ** attempt
                    st.warning(f"Rate limit hit. Waiting {wait}s…")
                    time.sleep(wait)
                    continue
                st.error(f"API error {resp.status_code}: {resp.text}")
                return None
            except requests.Timeout:
                st.warning(f"Timeout (attempt {attempt+1}/{max_retries})")
            except requests.RequestException as e:
                st.error(f"Network error: {e}")
                return None
        st.error("Failed after multiple retries")
        return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         Streamlit Helper Functions                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def load_api_key() -> str | None:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if api_key:
        return api_key
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        return st.text_input(
            "Enter your Perplexity API Key:",
            type="password",
            help="Create/find key at https://www.perplexity.ai/settings/api",
        )


def summarize_transcript(api: PerplexityAPI, transcript: str) -> str:
    if not transcript.strip():
        return "No transcript content to summarize."
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS] + "…"
        st.warning("Transcript truncated to fit context window.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert educational content analyst. "
                "Summarize thoroughly, capturing all key ideas, examples, and take-aways."
            ),
        },
        {
            "role": "user",
            "content": (
                "Please provide a comprehensive summary of the following lecture transcript. \n\n"
                f"{transcript}"
            ),
        },
    ]
    with st.spinner("🤖 Creating detailed summary…"):
        summary = api.chat_completion(messages)
    return summary or "Summary generation failed."


def find_learning_resources(api: PerplexityAPI, summary: str) -> str:
    """Ask Perplexity to suggest learning resources."""
    extract_topics = [
        {
            "role": "system",
            "content": "Extract 3-5 concise key topics from the content.",
        },
        {
            "role": "user",
            "content": summary[:1_000],
        },
    ]
    topics = api.chat_completion(extract_topics) or ""
    search_prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who recommends high-quality learning resources "
                "(courses, books, tutorials, docs, videos) for self-study."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Suggest the best resources to learn more about: {topics or 'the lecture topics'}.\n"
                "For each resource include a short description and a direct URL."
            ),
        },
    ]
    with st.spinner("🔍 Collecting learning resources…"):
        resources = api.chat_completion(search_prompt)
    return resources or "No resources found."


def save_temp_file(uploaded_file) -> str | None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        st.error(f"Failed to save temp file: {e}")
        return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                Main App                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    # ── Title & Intro ───────────────────────────────────────────────────────────
    st.title("🎧 Lecture Transcript Analyzer")
    st.markdown(
        """
Turn your audio/video lecture into a **detailed study summary** and a curated list of
**learning resources** – powered by Whisper + Perplexity (cheapest *sonar* model).
"""
    )

    # ── Session State defaults ─────────────────────────────────────────────────
    for key, default in {
        "show_results": False,
        "current_summary": "",
        "current_resources": "",
        "loaded_summary": "",
        "loaded_resources": "",
    }.items():
        st.session_state.setdefault(key, default)

    # ── API Key ────────────────────────────────────────────────────────────────
    api_key = load_api_key()
    if not api_key:
        st.info("Enter API key in sidebar to begin.")
        return
    api_client = PerplexityAPI(api_key)

    # ── Sidebar: Config & History ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Whisper Settings")
        model_choice = st.selectbox(
            "Model size", ["tiny", "base", "small", "medium"], index=1
        )
        chunk_seconds = st.slider(
            "Audio chunk duration (s)",
            min_value=60,
            max_value=600,
            value=300,
            step=30,
        )

        st.markdown("---")
        st.markdown("### 📚 Analysis History")
        history = load_history()
        if history:
            for item in reversed(history[-10:]):  # show last 10
                with st.expander(f"🎵 {item['filename'][:20]}…"):
                    st.write(f"**{item['timestamp']}**")
                    st.write(item["preview"])
                    if st.button("Load", key=item["id"]):
                        st.session_state["loaded_summary"] = item["summary"]
                        st.session_state["loaded_resources"] = item["resources"]
                        st.session_state["show_results"] = True
                        st.rerun()
        else:
            st.caption("No past analyses.")
        if st.button("🗑️ Clear history"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("History cleared.")
            st.rerun()

    # ── File Upload ────────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Audio/Video")
    uploaded_file = st.file_uploader(
        "Choose file",
        type=["mp3", "wav", "mp4", "avi", "mov", "mkv", "flv", "webm"],
    )

    # ── Analyze Button ─────────────────────────────────────────────────────────
    if st.button("🚀 Analyze", disabled=uploaded_file is None):
        if uploaded_file is None:
            st.warning("Please upload a file first.")
            st.stop()

        # Save file
        tmp_path = save_temp_file(uploaded_file)
        if not tmp_path:
            st.stop()

        tracker = ProgressTracker()
        tracker.initialize(total_steps=4)

        try:
            # Step 1: Whisper transcription
            tracker.update(1, "Initializing Whisper")
            setup_logging("INFO")
            transcriber = WhisperTranscriber(model_size=model_choice)

            tracker.update(2, "Transcribing audio")
            result = transcriber.transcribe_file(tmp_path, chunk_duration=chunk_seconds)
            os.unlink(tmp_path)
            transcriber.cleanup_temp_files()

            if not result.text.strip():
                st.error("No speech detected.")
                st.stop()

            # Step 2: Summarize
            tracker.update(3, "Generating summary")
            summary_text = summarize_transcript(api_client, result.text)

            # Step 3: Resources
            tracker.update(4, "Searching resources")
            resources_text = find_learning_resources(api_client, summary_text)

            tracker.complete()

            # Save & show results
            st.session_state["current_summary"] = summary_text
            st.session_state["current_resources"] = resources_text
            st.session_state["loaded_summary"] = ""
            st.session_state["loaded_resources"] = ""
            st.session_state["show_results"] = True

            save_to_history(summary_text, resources_text, uploaded_file.name)
            st.rerun()


        except Exception as e:
            tracker.complete()
            st.error(f"Processing failed: {e}")
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if "transcriber" in locals():
                transcriber.cleanup_temp_files()
            st.stop()

    # ── Display Results in Tabs ────────────────────────────────────────────────
    if st.session_state.get("show_results"):
        summary = (
            st.session_state["loaded_summary"] or st.session_state["current_summary"]
        )
        resources = (
            st.session_state["loaded_resources"] or st.session_state["current_resources"]
        )

        tab_sum, tab_cite = st.tabs(
            ["📝 Summary", f"🔗 Citations & Resources"]
        )

        with tab_sum:
            st.header("Detailed Summary")
            st.write(summary)
            st.download_button(
                "📥 Download Summary",
                data=summary,
                file_name=f"summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

        with tab_cite:
            st.header("Learning Resources")
            st.write(resources)

    # ── How-to Section ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
### 📖 How to Use

1. **Add API key** in the sidebar (Perplexity account required).  
2. **Upload** an audio/video lecture file.  
3. Adjust **Whisper settings** (model & chunk duration) if desired.  
4. Click **Analyze** and watch the progress loader with ETA.  
5. Explore results in the two tabs and revisit past runs in the sidebar.

*Tip *: Clear history anytime via the sidebar button.
"""
    )


if __name__ == "__main__":
    main()
 