"""Orchestrate complete RAG ingestion and question-answering workflows."""

from pathlib import Path

from src.cleaner import clean_text
from src.chunker import chunk_documents
from src.config import settings
from src.document_loader import iter_documents
from src.embeddings import GeminiEmbedder
from src.generator import GeminiGenerator, build_prompt
from src.vector_store import FaissVectorStore


def require_api_key() -> str:
    """Return the configured Google API key or raise a clear setup error."""

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to .env or your environment.")
    return settings.google_api_key


def ingest(data_dir: str, chunk_size: int, overlap: int) -> dict:
    """Parse, clean, chunk, embed, and persist all supported documents."""

    api_key = require_api_key()
    documents = [
        {
            "source": document["source"],
            "text": clean_text(document["text"]),
        }
        for document in iter_documents(data_dir)
    ]
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    embedder = GeminiEmbedder(api_key=api_key, model=settings.embedding_model)
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    store = FaissVectorStore.from_embeddings(chunks, embeddings)

    index_base = Path(settings.index_dir) / f"rag_chunks_{chunk_size}"
    store.save(index_base.with_suffix(".faiss"), index_base.with_suffix(".json"))
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "index_path": str(index_base.with_suffix(".faiss")),
        "metadata_path": str(index_base.with_suffix(".json")),
    }


def answer(question: str, chunk_size: int, top_k: int) -> dict:
    """Retrieve relevant chunks and return a Gemini answer with its sources."""

    api_key = require_api_key()
    index_base = Path(settings.index_dir) / f"rag_chunks_{chunk_size}"
    store = FaissVectorStore.load(index_base.with_suffix(".faiss"), index_base.with_suffix(".json"))

    embedder = GeminiEmbedder(api_key=api_key, model=settings.embedding_model)
    query_embedding = embedder.embed([question])[0]
    retrieved = store.search(query_embedding, top_k=top_k)

    prompt = build_prompt(question, retrieved)
    generator = GeminiGenerator(api_key=api_key, model=settings.generation_model)
    response = generator.generate(prompt)

    return {
        "answer": response,
        "sources": [
            {
                "source": item["chunk"].source,
                "chunk_index": item["chunk"].chunk_index,
                "chunk_size": item["chunk"].chunk_size,
                "similarity": item["score"],
                "text": item["chunk"].text,
            }
            for item in retrieved
        ],
        "prompt": prompt,
    }
