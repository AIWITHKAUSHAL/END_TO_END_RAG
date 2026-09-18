"""Streamlit UI for uploading, indexing, and querying RAG documents."""

from pathlib import Path
import uuid

import streamlit as st

from src.config import settings
from src.pipeline import answer, ingest


st.set_page_config(
    page_title="RAG knowledge studio",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "A transparent Retrieval-Augmented Generation workspace built "
            "with Gemini embeddings, FAISS retrieval, and source-aware answers."
        )
    },
)


def initialize_session() -> None:
    """Create the per-user state used by ingestion and answer display."""

    defaults = {
        "upload_session_id": uuid.uuid4().hex[:10],
        "index_chunk_size": None,
        "ingestion_result": None,
        "last_answer": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def save_uploaded_files(uploaded_files) -> Path:
    """Save sanitized uploads in a session-specific directory and return it."""

    upload_dir = Path("data/uploaded_docs") / st.session_state.upload_session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        (upload_dir / safe_name).write_bytes(uploaded_file.getbuffer())

    return upload_dir


def index_is_available(chunk_size: int) -> bool:
    """Return whether both persisted index files exist for a chunk size."""

    index_base = Path(settings.index_dir) / f"rag_chunks_{chunk_size}"
    return index_base.with_suffix(".faiss").is_file() and index_base.with_suffix(".json").is_file()


def render_answer(result: dict) -> None:
    """Render a generated answer, its ranked evidence, and the full prompt."""

    st.header("Grounded answer", icon=":material/auto_awesome:")
    with st.container(border=True):
        st.markdown(result["answer"])

    st.subheader("Retrieved evidence", icon=":material/source:")
    st.caption("Open a source to inspect the exact passage supplied to Gemini.")
    for rank, source in enumerate(result["sources"], start=1):
        source_name = Path(source["source"]).name
        label = (
            f"{rank}. {source_name} · chunk {source['chunk_index']} · "
            f"similarity {source['similarity']:.4f}"
        )
        with st.expander(label, icon=":material/article:"):
            st.caption(f"Full source path: {source['source']}")
            st.markdown(source["text"])

    with st.expander("Prompt sent to Gemini", icon=":material/code:"):
        st.caption("This is the complete question and retrieved context used for generation.")
        st.code(result["prompt"], language="text", wrap_lines=True)


initialize_session()
chunk_options = [250, 500, 900]
default_chunk_size = settings.default_chunk_size if settings.default_chunk_size in chunk_options else 500

with st.sidebar:
    st.header("Retrieval settings", icon=":material/tune:")
    chunk_size = st.segmented_control(
        "Chunk size",
        options=chunk_options,
        default=default_chunk_size,
        required=True,
        width="stretch",
        help="Maximum words per chunk. Smaller chunks are precise; larger chunks retain more context.",
    )
    overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=200,
        value=min(settings.default_chunk_overlap, 200),
        step=10,
        help="Words repeated between neighboring chunks to preserve context at boundaries.",
    )
    top_k = st.slider(
        "Evidence chunks",
        min_value=1,
        max_value=8,
        value=min(max(settings.top_k, 1), 8),
        help="Number of similar chunks included in the generation prompt.",
    )

    st.subheader("Active models", icon=":material/memory:")
    st.caption(f"Generation · `{settings.generation_model}`")
    st.caption(f"Embeddings · `{settings.embedding_model}`")
    st.caption("Indexes are stored locally; Gemini calls require an internet connection.")

st.title("RAG knowledge studio", icon=":material/hub:")
st.markdown(
    "Turn your documents into a searchable knowledge base, retrieve the strongest evidence, "
    "and generate answers you can trace back to the source."
)

with st.container(horizontal=True, gap="small"):
    st.badge("PDF, TXT, MD", icon=":material/description:", color="blue")
    st.badge("FAISS retrieval", icon=":material/database:", color="violet")
    st.badge("Source-aware answers", icon=":material/fact_check:", color="green")

ingest_panel, question_panel = st.columns(2, gap="large")

with ingest_panel.container(border=True, height="stretch"):
    st.header("1. Build the knowledge base", icon=":material/upload_file:")
    st.caption("Upload documents, choose retrieval settings, and create a local vector index.")
    uploaded_files = st.file_uploader(
        "Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, plain text, and Markdown.",
    )

    if uploaded_files:
        total_bytes = sum(upload.size for upload in uploaded_files)
        st.caption(f"{len(uploaded_files)} file(s) selected · {total_bytes / 1024:.1f} KB total")
    else:
        st.caption("Select at least one document to enable ingestion.")

    ingest_clicked = st.button(
        "Create vector index",
        type="primary",
        icon=":material/database_upload:",
        disabled=not uploaded_files,
        width="stretch",
    )

    if ingest_clicked:
        with st.status("Building the knowledge base…", expanded=True) as status:
            try:
                st.write("Saving and parsing uploaded documents")
                data_dir = save_uploaded_files(uploaded_files)
                st.write("Cleaning text and creating overlapping chunks")
                st.write("Creating Gemini embeddings and writing the FAISS index")
                result = ingest(str(data_dir), chunk_size, overlap)
            except Exception as exc:
                status.update(label="Ingestion failed", state="error", expanded=True)
                st.error(str(exc), icon=":material/error:")
            else:
                st.session_state.index_chunk_size = chunk_size
                st.session_state.ingestion_result = result
                st.session_state.last_answer = None
                status.update(label="Knowledge base ready", state="complete", expanded=False)
                st.toast("Documents indexed successfully", icon=":material/check_circle:")

    if st.session_state.ingestion_result:
        result = st.session_state.ingestion_result
        metric_columns = st.columns(2)
        metric_columns[0].metric("Documents", result["documents"], border=True)
        metric_columns[1].metric("Chunks", result["chunks"], border=True)
        st.caption(f"Index: `{result['index_path']}`")

active_chunk_size = st.session_state.index_chunk_size or chunk_size
index_available = index_is_available(active_chunk_size)

with question_panel.container(border=True, height="stretch"):
    st.header("2. Ask your documents", icon=":material/question_answer:")
    st.caption("Your question is matched against the index before Gemini writes an answer.")

    if index_available:
        st.success(
            f"Index ready · {active_chunk_size}-word chunks",
            icon=":material/check_circle:",
        )
    else:
        st.info(
            "Create an index first, or select a chunk size with an existing index.",
            icon=":material/info:",
        )

    with st.form("question_form", border=False):
        question = st.text_area(
            "Question",
            placeholder="What are the main stages in a RAG pipeline?",
            height=120,
            help="Ask a focused question that can be answered from the indexed documents.",
        )
        ask_clicked = st.form_submit_button(
            "Find evidence and answer",
            type="primary",
            icon=":material/search:",
            disabled=not index_available,
            width="stretch",
        )

    st.caption("Example: “Why is returning provenance important in a RAG system?”")

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question before searching.", icon=":material/warning:")
        else:
            with st.status("Retrieving evidence and generating an answer…") as status:
                try:
                    result = answer(question.strip(), active_chunk_size, top_k)
                except Exception as exc:
                    status.update(label="Question answering failed", state="error", expanded=True)
                    st.error(str(exc), icon=":material/error:")
                else:
                    st.session_state.last_answer = result
                    status.update(label="Answer ready", state="complete", expanded=False)

if st.session_state.last_answer:
    render_answer(st.session_state.last_answer)
else:
    st.info(
        "Your answer and its source evidence will appear here after you submit a question.",
        icon=":material/lightbulb:",
    )
