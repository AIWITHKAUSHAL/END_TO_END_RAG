from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    chunk_size: int
    overlap: int


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_id = f"{source}::chunk-{index}::size-{chunk_size}"
        chunks.append(
            Chunk(
                id=chunk_id,
                text=" ".join(chunk_words),
                source=source,
                chunk_index=index,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
        if end == len(words):
            break
        start = end - overlap
        index += 1

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(
            chunk_text(
                text=document["text"],
                source=document["source"],
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return all_chunks
