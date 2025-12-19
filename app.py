"""Streamlit UI for LangGraph RAG Agent.

A professional interface for interacting with the LangGraph agent,
managing documents, and monitoring RAG ingestion.
"""

import asyncio
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the LangGraph agent
from react_agent import graph
from react_agent.context import Context

# Constants
DATA_DIR = Path(__file__).parent / "src" / "data"
INGESTION_SCRIPT = Path(__file__).parent / "src" / "rag" / "ingestion.py"
TRACKING_FILE = Path(__file__).parent / "processed_documents.json"
MAX_FILE_SIZE_MB = 3
MAX_FILES_UPLOAD = 2
QDRANT_COLLECTION_NAME = "document_chunks"

# Page config
st.set_page_config(
    page_title="LangGraph RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-dark: #1e1b4b;
        --bg-light: #f8fafc;
    }

    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* File card styling */
    .file-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    .file-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }

    .file-name {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
    }

    .file-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.2);
    }

    .assistant-message {
        background: white;
        color: #1e293b;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-success {
        background: #d1fae5;
        color: #065f46;
    }

    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }

    .status-info {
        background: #dbeafe;
        color: #1e40af;
    }

    /* Terminal output styling */
    .terminal-output {
        background: #1e293b;
        color: #10b981;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Stats card */
    .stats-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #6366f1;
    }

    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }

    .stats-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingestion_running" not in st.session_state:
        st.session_state.ingestion_running = False
    if "ingestion_output" not in st.session_state:
        st.session_state.ingestion_output = []
    if "show_ingestion_logs" not in st.session_state:
        st.session_state.show_ingestion_logs = False
    if "ingestion_success" not in st.session_state:
        st.session_state.ingestion_success = None
    if "ingestion_log_output" not in st.session_state:
        st.session_state.ingestion_log_output = []


def get_files_in_data_dir() -> List[dict]:
    """Get list of files in the data directory with metadata."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []

    files = []
    for file_path in DATA_DIR.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "path": file_path,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
            })

    return sorted(files, key=lambda x: x["modified"], reverse=True)


def get_processed_documents_stats() -> dict:
    """Get statistics from tracking file about processed documents."""
    try:
        if not TRACKING_FILE.exists():
            return {"count": 0, "total_chunks": 0}

        import json
        with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
            tracking_data = json.load(f)

        processed_docs = tracking_data.get("processed_documents", {})
        count = len(processed_docs)
        total_chunks = sum(doc["chunks_count"] for doc in processed_docs.values())

        return {"count": count, "total_chunks": total_chunks}
    except:
        return {"count": 0, "total_chunks": 0}


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def delete_file(file_path: Path) -> bool:
    """Delete a file from the data directory."""
    try:
        file_path.unlink()
        return True
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False


def clear_processing_history() -> tuple[bool, str]:
    """Clear the processing history (tracking JSON file)."""
    try:
        if TRACKING_FILE.exists():
            TRACKING_FILE.unlink()
            return True, "Successfully cleared processing history"
        else:
            return True, "No processing history to clear"

    except Exception as e:
        return False, f"Error clearing processing history: {str(e)}"


def clear_vector_store() -> tuple[bool, str]:
    """Clear all vectors from the Qdrant collection."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        # Get Qdrant credentials from environment
        qdrant_url = os.getenv("QDRANT_URL", "https://f58f1067-58c2-413d-acd3-8e6058389e20.us-east4-0.gcp.cloud.qdrant.io")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.UvgYpKQ7MmBOsYDTXxSx-Pg_g5l0Ti-oDxgqG2I80SQ")

        # Connect to Qdrant
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=True,
        )

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if QDRANT_COLLECTION_NAME not in collection_names:
            return False, f"Collection '{QDRANT_COLLECTION_NAME}' not found"

        # Get point count before deletion
        info_before = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        points_count = info_before.points_count

        if points_count == 0:
            return True, "Vector store is already empty"

        # Delete all points by scrolling through IDs
        offset = None
        batch_size = 100
        total_deleted = 0

        while True:
            # Scroll through points to get their IDs
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )

            if not points:
                break

            # Extract IDs
            point_ids = [point.id for point in points]

            # Delete points by ID
            client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )

            total_deleted += len(point_ids)

            if offset is None:
                break

        return True, f"Successfully deleted {total_deleted} vectors from collection '{QDRANT_COLLECTION_NAME}'"

    except Exception as e:
        return False, f"Error clearing vector store: {str(e)}"


def run_ingestion_script(log_container):
    """Run the ingestion script and capture output in real-time.

    Args:
        log_container: Streamlit container to display logs in real-time
    """
    st.session_state.ingestion_running = True
    st.session_state.ingestion_output = []

    try:
        # Get the project root directory
        project_root = Path(__file__).parent

        # Prepare environment variables - copy current env and ensure all are passed
        env = os.environ.copy()

        # Run the ingestion script with proper environment and working directory
        # Set PYTHONIOENCODING to handle emojis and unicode characters on Windows
        env['PYTHONIOENCODING'] = 'utf-8'
        # Disable Python output buffering for immediate output
        env['PYTHONUNBUFFERED'] = '1'

        process = subprocess.Popen(
            [sys.executable, "-u", str(INGESTION_SCRIPT), "--yes"],  # -u for unbuffered
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr separately to see errors
            text=True,
            bufsize=0,  # Unbuffered
            universal_newlines=True,
            env=env,
            cwd=str(project_root),
            encoding='utf-8',
            errors='replace',
        )

        # Capture output line by line in real-time
        output_lines = []
        error_lines = []

        # Create a text area in the log container to show real-time output
        with log_container:
            log_display = st.empty()
            status_display = st.empty()

            import select
            import time

            # Read stdout and stderr in real-time
            while True:
                # Read stdout
                stdout_line = process.stdout.readline()
                if stdout_line:
                    line = stdout_line.rstrip()
                    if line:
                        output_lines.append(line)
                        st.session_state.ingestion_output.append(line)
                        # Update the display in real-time
                        log_display.code('\n'.join(output_lines[-50:]), language='bash')  # Show last 50 lines

                # Read stderr
                stderr_line = process.stderr.readline()
                if stderr_line:
                    line = stderr_line.rstrip()
                    if line:
                        error_lines.append(f"[ERROR] {line}")
                        output_lines.append(f"[ERROR] {line}")
                        # Update the display in real-time
                        log_display.code('\n'.join(output_lines[-50:]), language='bash')

                # Check if process has finished
                if process.poll() is not None:
                    # Read any remaining output
                    remaining_stdout = process.stdout.read()
                    if remaining_stdout:
                        for line in remaining_stdout.split('\n'):
                            if line.strip():
                                output_lines.append(line.strip())

                    remaining_stderr = process.stderr.read()
                    if remaining_stderr:
                        for line in remaining_stderr.split('\n'):
                            if line.strip():
                                error_lines.append(f"[ERROR] {line.strip()}")
                                output_lines.append(f"[ERROR] {line.strip()}")

                    break

                # Small delay to prevent CPU spinning
                time.sleep(0.01)

            # Final display update
            log_display.code('\n'.join(output_lines), language='bash')
            status_display.info(f"Process finished with return code: {process.returncode}")

        st.session_state.ingestion_running = False

        # Combine output and errors
        all_output = output_lines + error_lines

        if process.returncode == 0:
            return True, all_output if all_output else ["Ingestion completed successfully (no output)"]
        else:
            return False, all_output if all_output else [f"Process failed with return code: {process.returncode}"]

    except Exception as e:
        st.session_state.ingestion_running = False
        import traceback
        error_details = traceback.format_exc()
        error_lines = [f"EXCEPTION: {str(e)}", "Full traceback:", error_details]

        # Show error in log container
        with log_container:
            st.error('\n'.join(error_lines))

        return False, error_lines


async def chat_with_agent(user_message: str, context: Context):
    """Send a message to the LangGraph agent and get response."""
    try:
        # Invoke the graph
        result = await graph.ainvoke(
            {"messages": [("user", user_message)]},
            context=context,
        )

        # Extract the final message
        if result and "messages" in result and len(result["messages"]) > 0:
            final_message = result["messages"][-1]
            return final_message.content
        else:
            return "No response received from agent."

    except Exception as e:
        return f"Error: {str(e)}"


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 LangGraph RAG Agent</h1>
        <p>Intelligent document search and conversational AI</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with file management."""
    with st.sidebar:
        st.markdown("### 📁 Document Management")

        # Get current data
        files = get_files_in_data_dir()
        stats = get_processed_documents_stats()

        # Stats row
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(files)}</div>
                <div class="stats-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{stats["count"]}</div>
                <div class="stats-label">Processed</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{stats["total_chunks"]}</div>
                <div class="stats-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # File upload section
        st.markdown("#### ⬆️ Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["pdf", "txt", "doc", "docx", "md"],
            help=f"Maximum {MAX_FILES_UPLOAD} files, {MAX_FILE_SIZE_MB}MB each",
            key="file_uploader",
        )

        if uploaded_files:
            if len(uploaded_files) > MAX_FILES_UPLOAD:
                st.error(f"⚠️ Maximum {MAX_FILES_UPLOAD} files allowed at once!")
            else:
                valid_files = []
                for uploaded_file in uploaded_files:
                    size_mb = uploaded_file.size / (1024 * 1024)
                    if size_mb > MAX_FILE_SIZE_MB:
                        st.error(f"⚠️ {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit!")
                    else:
                        valid_files.append(uploaded_file)

                if valid_files and st.button("💾 Save Files", type="primary"):
                    for uploaded_file in valid_files:
                        file_path = DATA_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    st.success(f"✅ Saved {len(valid_files)} file(s)!")
                    st.rerun()

        st.markdown("---")

        # Pipeline Controls
        st.markdown("#### 🔄 Pipeline Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            parse_clicked = st.button("🚀\nParse", disabled=st.session_state.ingestion_running, help="Parse and ingest documents", use_container_width=True)

        # Handle parse button click outside the column to use full width
        if parse_clicked:
            if len(files) == 0:
                st.warning("⚠️ No files to parse!")
            elif not INGESTION_SCRIPT.exists():
                st.error(f"❌ Ingestion script not found at: {INGESTION_SCRIPT}")
            else:
                # Create a container for real-time log output
                st.markdown("---")
                st.markdown("### 📋 Ingestion Progress")
                st.markdown("*Live output from ingestion script*")

                # Create log container that will be updated in real-time
                log_container = st.container()

                try:
                    # Run the ingestion with real-time output
                    success, output = run_ingestion_script(log_container)

                    # Save to session state to persist logs
                    st.session_state.show_ingestion_logs = True
                    st.session_state.ingestion_success = success
                    st.session_state.ingestion_log_output = output

                    # Show results after completion
                    st.markdown("---")
                    if success:
                        st.success("✅ Ingestion completed successfully!")
                        st.markdown(f"**Processed:** {len(output)} log lines")
                        st.balloons()
                    else:
                        st.error("❌ Ingestion failed! Check the logs above for details.")
                        st.markdown("**Tip:** Look for `[ERROR]` lines in the log above")
                        st.markdown("**Common issues:**")
                        st.markdown("- API key issues (LLMWhisperer, NVIDIA, Qdrant)")
                        st.markdown("- Network connectivity problems")
                        st.markdown("- Invalid or corrupted documents")

                    # Add close button
                    col_close1, col_close2, col_close3 = st.columns([1, 1, 1])
                    with col_close2:
                        if st.button("✖️ Close Logs & Refresh", type="primary", use_container_width=True):
                            st.session_state.show_ingestion_logs = False
                            st.session_state.ingestion_success = None
                            st.session_state.ingestion_log_output = []
                            st.rerun()

                except Exception as e:
                    st.error(f"❌ Critical error: {str(e)}")
                    import traceback
                    with st.expander("🐛 Debug Info"):
                        st.code(traceback.format_exc())

                    # Add close button for errors too
                    if st.button("✖️ Close Error Log"):
                        st.rerun()

        # Show persistent logs if they exist
        elif st.session_state.show_ingestion_logs:
            st.markdown("---")
            st.markdown("### 📋 Ingestion Logs (Previous Run)")

            # Show the saved logs
            if st.session_state.ingestion_log_output:
                st.code('\n'.join(st.session_state.ingestion_log_output), language='bash')

            # Show status
            st.markdown("---")
            if st.session_state.ingestion_success:
                st.success("✅ Ingestion completed successfully!")
            else:
                st.error("❌ Ingestion failed! Check the logs above for details.")

            # Close button
            col_close1, col_close2, col_close3 = st.columns([1, 1, 1])
            with col_close2:
                if st.button("✖️ Close Logs & Refresh", type="primary", use_container_width=True, key="close_persistent"):
                    st.session_state.show_ingestion_logs = False
                    st.session_state.ingestion_success = None
                    st.session_state.ingestion_log_output = []
                    st.rerun()

        with col2:
            if st.button("🧹\nClear\nHistory", help="Clear processing history", use_container_width=True):
                if stats["count"] == 0:
                    st.info("No processing history to clear")
                else:
                    with st.spinner("Clearing..."):
                        success, message = clear_processing_history()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                    st.rerun()

        with col3:
            if st.button("💥\nClear\nVectors", help="Clear vector store", use_container_width=True):
                with st.spinner("Clearing..."):
                    success, message = clear_vector_store()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                st.rerun()

        st.markdown("---")

        # Documents list (collapsible)
        with st.expander(f"📄 Documents ({len(files)})", expanded=len(files) > 0 and len(files) <= 3):
            if len(files) == 0:
                st.info("No documents uploaded")
            else:
                for file_info in files:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="file-card">
                            <div class="file-name">📄 {file_info['name']}</div>
                            <div class="file-meta">{format_file_size(file_info['size'])}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button("🗑️", key=f"delete_{file_info['name']}", help="Delete"):
                            if delete_file(file_info['path']):
                                st.success("Deleted!")
                                st.rerun()

        st.markdown("---")

        # Settings
        st.markdown("#### ⚙️ Settings")

        with st.expander("LLM Configuration"):
            model_name = st.text_input(
                "Model",
                value="groq/openai/gpt-oss-120b",
                help="Format: provider/model-name",
            )
            st.session_state.model_name = model_name


def render_chat_interface():
    """Render the main chat interface."""
    st.markdown("### 💬 Chat with Your Documents")

    # Display chat messages
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong><br>{content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Agent:</strong><br>{content}
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    st.markdown("---")

    col1, col2 = st.columns([6, 1])

    with col1:
        user_input = st.text_input(
            "Your message:",
            placeholder="Ask a question about your documents or the weather...",
            key="user_input",
            label_visibility="collapsed",
        )

    with col2:
        send_button = st.button("Send 📤", type="primary", use_container_width=True)

    # Process message
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get agent response
        with st.spinner("🤔 Agent is thinking..."):
            context = Context(
                model=st.session_state.get("model_name", "groq/openai/gpt-oss-120b")
            )

            response = asyncio.run(chat_with_agent(user_input, context))

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Rerun to update UI
        st.rerun()

    # Clear chat button
    if len(st.session_state.messages) > 0:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def main():
    """Main application."""
    initialize_session_state()

    render_header()
    render_sidebar()

    # Main content area
    render_chat_interface()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
        <p>Powered by LangGraph 🦜 | Built with Streamlit 🎈</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
