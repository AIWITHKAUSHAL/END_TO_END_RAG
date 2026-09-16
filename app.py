from pathlib import Path
import uuid

import streamlit as st

from src.config import settings
from src.pipeline import answer, ingest


st.set_page_config(page_title="End-to-End RAG", page_icon="RAG", layout="wide")

st.title("End-to-End RAG Application")


def save_uploaded_files(uploaded_files) -> Path:
    if "upload_session_id" not in st.session_state:
        st.session_state.upload_session_id = uuid.uuid4().hex[:10]

    upload_dir = Path("data/uploaded_docs") / st.session_state.upload_session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        output_path = upload_dir / safe_name
        output_path.write_bytes(uploaded_file.getbuffer())

    return upload_dir


with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    chunk_size = st.selectbox("Chunk size", [250, 500, 900], index=1)
    overlap = st.slider("Chunk overlap", 0, 200, settings.default_chunk_overlap, 10)
    top_k = st.slider("Top-K chunks", 1, 8, settings.top_k)
    st.caption(f"Generator: {settings.generation_model}")
    st.caption(f"Embeddings: {settings.embedding_model}")

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) ready for ingestion.")
    else:
        st.info("Upload at least one PDF, TXT, or MD file.")

    if st.button("Ingest uploaded documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Parsing, cleaning, chunking, embedding, and indexing..."):
            try:
                data_dir = save_uploaded_files(uploaded_files)
                result = ingest(str(data_dir), chunk_size, overlap)
                st.session_state.index_chunk_size = chunk_size
                st.success(f"Indexed {result['chunks']} chunks from {result['documents']} documents.")
                st.json(result)
            except Exception as exc:
                st.error(str(exc))

question = st.text_input("Ask a question about the indexed documents")

if st.button("Ask", disabled=not question):
    with st.spinner("Embedding query, searching FAISS, and generating answer..."):
        try:
            indexed_chunk_size = st.session_state.get("index_chunk_size", chunk_size)
            result = answer(question, indexed_chunk_size, top_k)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            for rank, source in enumerate(result["sources"], start=1):
                with st.expander(
                    f"{rank}. {source['source']} | chunk {source['chunk_index']} | score {source['similarity']:.4f}"
                ):
                    st.write(source["text"])

            with st.expander("Prompt sent to Gemini"):
                st.code(result["prompt"])
