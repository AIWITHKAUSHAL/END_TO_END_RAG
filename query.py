"""Command-line entry point for asking a question against a RAG index."""

import argparse

from rich.console import Console
from rich.table import Table

from src.config import settings
from src.pipeline import answer


console = Console()


def main() -> None:
    """Parse a question, run retrieval and generation, and print cited sources."""

    parser = argparse.ArgumentParser(description="Ask a question using the local FAISS RAG index.")
    parser.add_argument("question")
    parser.add_argument("--chunk-size", type=int, default=settings.default_chunk_size)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    result = answer(args.question, args.chunk_size, args.top_k)
    console.rule("Answer")
    console.print(result["answer"])

    table = Table(title="Sources Used")
    table.add_column("Rank")
    table.add_column("Source")
    table.add_column("Chunk")
    table.add_column("Similarity")
    for rank, source in enumerate(result["sources"], start=1):
        table.add_row(
            str(rank),
            source["source"],
            str(source["chunk_index"]),
            f"{source['similarity']:.4f}",
        )
    console.print(table)


if __name__ == "__main__":
    main()
