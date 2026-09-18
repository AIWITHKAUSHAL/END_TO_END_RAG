"""Build, persist, load, and search a cosine-similarity FAISS index."""

from pathlib import Path
import json

import faiss
import numpy as np

from src.chunker import Chunk


class FaissVectorStore:
    """Keep a FAISS vector index aligned with its source-aware text chunks."""

    def __init__(self, index: faiss.IndexFlatIP, chunks: list[Chunk]):
        """Pair an existing FAISS index with chunks in matching row order."""

        self.index = index
        self.chunks = chunks

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vector rows so inner product represents cosine similarity."""

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    @classmethod
    def from_embeddings(cls, chunks: list[Chunk], embeddings: list[list[float]]) -> "FaissVectorStore":
        """Create a normalized inner-product index from chunks and embeddings."""

        vectors = np.array(embeddings, dtype="float32")
        vectors = cls._normalize(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, chunks=chunks)

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Write the FAISS index and JSON-serialized chunk metadata to disk."""

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata = [chunk.__dict__ for chunk in self.chunks]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> "FaissVectorStore":
        """Restore an index and its ordered chunk metadata from disk."""

        index = faiss.read_index(str(index_path))
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in metadata]
        return cls(index=index, chunks=chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Return up to Top-K nearest chunks with their similarity scores."""

        query = np.array([query_embedding], dtype="float32")
        query = self._normalize(query)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "score": float(score),
                    "chunk": chunk,
                }
            )
        return results
