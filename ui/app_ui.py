import os
import time
import requests
import streamlit as st

# --- Configuration & Styling ---
st.set_page_config(
    page_title="RepoMind AI - GitHub RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base URL for FastAPI Backend (Supports environment override)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_data(ttl=5)
def check_api_health():
    """Verify backend API connection with caching to prevent high-frequency overhead."""
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def get_latest_repo():
    """Fetch the most recently completed repository from the backend."""
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/repos/latest", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def create_auto_session():
    """Fetch an automatic UUID session from FastAPI bound to the latest completed repo."""
    try:
        res = requests.post(f"{API_BASE_URL}/api/v1/sessions/auto", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def ingest_repo(repo_url: str):
    """Trigger repository ingestion via FastAPI."""
    try:
        res = requests.post(
            f"{API_BASE_URL}/api/v1/repos/ingest",
            json={"repo_url": repo_url},
            timeout=(5, 15)
        )
        return res.json(), res.status_code
    except requests.exceptions.Timeout:
        return {"error": "Ingestion submission timed out."}, 504
    except Exception as e:
        return {"error": str(e)}, 500


def fetch_repo_status(repo_id: str):
    """Fetch ingestion status for a specific repository."""
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/repos/{repo_id}", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def stream_chat_response(session_id: str, query: str):
    """Safely stream AI response from FastAPI with connection keep-alive."""
    session = requests.Session()
    try:
        with session.post(
            f"{API_BASE_URL}/api/v1/chat/stream",
            json={"session_id": session_id, "query": query},
            stream=True,
            timeout=(5, 60)
        ) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                    if chunk:
                        yield chunk
            else:
                yield f"❌ **Error ({response.status_code}):** {response.text}"
    except requests.exceptions.ConnectionError:
        yield "❌ **Connection Error:** Failed to connect to FastAPI backend."
    except requests.exceptions.Timeout:
        yield "❌ **Timeout Error:** Response stream timed out."
    except Exception as e:
        yield f"❌ **Stream Error:** {str(e)}"
    finally:
        session.close()


def reset_chat_state():
    """Completely reset the active session state so UI binds to the newest repo."""
    st.session_state.session_id = None
    st.session_state.repo_id = None
    st.session_state.repo_name = None
    st.session_state.messages = []


# --- App State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state or not st.session_state.session_id:
    st.session_state.session_id = None
    st.session_state.repo_id = None
    st.session_state.repo_name = None


def ensure_latest_repo_session():
    """Ensure the active chat session is bound to the latest completed repo."""
    # Handle a repo ingested during this UI session
    pending = st.session_state.get("pending_repo_id")
    if pending:
        pending_info = fetch_repo_status(pending)
        if pending_info:
            pending_status = pending_info.get("status", "")
            if pending_status == "COMPLETED":
                st.session_state.pending_repo_id = None
                # Force a fresh session so chat binds to the newly ingested repo
                st.session_state.session_id = None
                st.session_state.repo_id = None
                st.session_state.repo_name = None
                st.session_state.messages = []
                st.info(f"📦 New repo **{pending_info.get('repo_name')}** is indexed — chat switched to it.")
            elif pending_status.startswith("FAILED"):
                st.session_state.pending_repo_id = None
                st.error(f"❌ Latest ingestion failed: {pending_status}")
            else:
                st.warning(f"⏳ New repository `{pending_info.get('repo_name')}` is still processing... Answers below come from the previously indexed repo until it's ready.")
        else:
            st.warning("Unable to check ingestion status of the latest repo.")

    latest = get_latest_repo()
    if not latest:
        return None, False

    current_repo_id = st.session_state.get("repo_id")

    # If session is already bound to the latest repo, keep it
    if st.session_state.get("session_id") and current_repo_id == latest["repo_id"]:
        return latest, False

    # Otherwise fetch new auto session from backend
    session = create_auto_session()
    if session and session.get("session_id"):
        st.session_state.session_id = session["session_id"]
        st.session_state.repo_id = session.get("repo_id")
        st.session_state.repo_name = session.get("repo_name", latest.get("repo_name", "Ingested Repo"))
        st.session_state.messages = []
        return latest, True

    return latest, False


# --- Sidebar Navigation ---
st.sidebar.title("⚡ RepoMind AI")
st.sidebar.markdown("---")

api_online = check_api_health()
if api_online:
    st.sidebar.success("Backend Connected")
else:
    st.sidebar.error("Backend Disconnected (Check FastAPI port 8000)")

page = st.sidebar.radio("Navigation", ["📥 Ingest Repository", "💬 Codebase Chat"])


# ==========================================
# PAGE 1: REPOSITORY INGESTION MANAGER
# ==========================================
if page == "📥 Ingest Repository":
    st.header("📥 GitHub Repository Ingestion")
    st.write("Submit a public GitHub repository to clone, chunk, embed, and store in vector memory.")

    col1, col2 = st.columns([3, 1])
    
    with col1:
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/tiangolo/fastapi",
            help="Enter full HTTPS URL of a public repository."
        )

    with col2:
        st.write(" ")
        st.write(" ")
        submit_btn = st.button("Start Ingestion", type="primary", use_container_width=True)

    if submit_btn:
        if not repo_url.strip():
            st.warning("Please enter a valid GitHub repository URL.")
        elif not api_online:
            st.error("Cannot connect to backend server. Ensure FastAPI is running.")
        else:
            with st.spinner("Submitting repository for ingestion..."):
                result, status = ingest_repo(repo_url)
                
                if status == 200:
                    repo_id = result.get("repo_id")
                    st.session_state.pending_repo_id = repo_id

                    st.success("✅ Ingestion Triggered!")
                    st.json(result)

                    # Poll backend until ingestion completes (or times out)
                    progress = st.progress(0.0)
                    status_box = st.empty()
                    outcome = None
                    for i in range(20):
                        time.sleep(5)
                        info = fetch_repo_status(repo_id) if repo_id else None
                        if info:
                            cur_status = info.get("status", "PROCESSING")
                            status_box.info(f"⏳ Status: `{cur_status}` · files: `{info.get('total_files', '?')}`")
                            progress.progress(min((i + 1) / 20, 1.0))
                            if cur_status == "COMPLETED":
                                outcome = "completed"
                                break
                            if cur_status.startswith("FAILED"):
                                outcome = cur_status
                                break
                        else:
                            status_box.warning("Backend unreachable while polling, retrying...")

                    if outcome == "completed":
                        st.session_state.pending_repo_id = None
                        reset_chat_state()
                        st.success(f"✅ Repository indexed with `{info.get('total_files')}` files. Switch to **💬 Codebase Chat** — it is now bound to this repo.")
                    elif outcome:
                        st.session_state.pending_repo_id = None
                        st.error(f"❌ Ingestion failed: `{outcome}`")
                    else:
                        st.info("⏳ Still processing... The **💬 Codebase Chat** page will automatically switch to this repo once it's ready.")
                else:
                    st.error(f"Failed to submit repository: {result}")

    st.markdown("---")
    st.subheader("💡 How Ingestion Works")
    st.markdown("""
    1. **Clone & Parse:** Clones the repository and strips out binary/non-code files.
    2. **Chunking:** Segments source code into semantic context blocks using Tree-Sitter / Language Parsers.
    3. **Embedding:** Generates vector embeddings using local models / Hugging Face.
    4. **Storage:** Indexes vectors into ChromaDB with strict `repo_url` metadata filtering.
    """)


# ==========================================
# PAGE 2: INTERACTIVE RAG CHAT INTERFACE
# ==========================================
elif page == "💬 Codebase Chat":
    st.header("💬 Chat with Codebase")

    latest_repo, repo_switched = ensure_latest_repo_session()

    if not latest_repo:
        st.warning("No completed repository found yet. Ingest a repository on the **📥 Ingest Repository** page first.")
    else:
        if repo_switched:
            repo_display = st.session_state.get("repo_name") or latest_repo.get("repo_name", "New Repo")
            st.info(f"📦 Active repository updated — now chatting with **{repo_display}**")

        # Header Control Bar
        col_info, col_btn1, col_btn2 = st.columns([3, 1, 1])
        with col_info:
            repo_label = st.session_state.get("repo_name") or latest_repo.get("repo_name", "Active Repo")
            st.caption(f"📦 Repository: `{repo_label}` · Session ID: `{st.session_state.session_id}`")
        with col_btn1:
            if st.button("🔄 New Thread", use_container_width=True):
                session = create_auto_session()
                if session and session.get("session_id"):
                    st.session_state.session_id = session["session_id"]
                    st.session_state.repo_id = session.get("repo_id")
                    st.session_state.repo_name = session.get("repo_name")
                st.session_state.messages = []
                st.rerun()
        with col_btn2:
            if st.button("🧹 Clear Screen", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")

        # Render Message History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User Direct Query Input
        if query := st.chat_input("Ask anything about the repository code..."):
            # Display User Message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Display Assistant Streamed Message
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Consume streaming generator from FastAPI endpoint
                for chunk in stream_chat_response(st.session_state.session_id, query):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)

            # Store response in session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})