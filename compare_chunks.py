"""Compare retrieval results produced by multiple document chunk sizes."""

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.pipeline import answer, ingest


console = Console()


def main() -> None:
    """Build each requested index and save ranked retrieval results to CSV."""

    parser = argparse.ArgumentParser(description="Compare retrieval results for three chunk sizes.")
    parser.add_argument("--question", default="What are the main stages in a RAG pipeline?")
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[250, 500, 900])
    parser.add_argument("--overlap", type=int, default=settings.default_chunk_overlap)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for chunk_size in args.chunk_sizes:
        console.print(f"[bold]Building index for chunk size {chunk_size}[/bold]")
        ingest(args.data_dir, chunk_size, args.overlap)
        result = answer(args.question, chunk_size, args.top_k)

        for rank, source in enumerate(result["sources"], start=1):
            rows.append(
                {
                    "chunk_size": chunk_size,
                    "rank": rank,
                    "similarity": round(source["similarity"], 4),
                    "source": source["source"],
                    "chunk_index": source["chunk_index"],
                    "preview": source["text"][:180].replace("\n", " "),
                }
            )

    output_path = Path("chunk_comparison_results.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)

    table = Table(title="Chunk Size Retrieval Comparison")
    for column in ["chunk_size", "rank", "similarity", "source", "chunk_index", "preview"]:
        table.add_column(column)
    for row in rows:
        table.add_row(*(str(row[column]) for column in ["chunk_size", "rank", "similarity", "source", "chunk_index", "preview"]))
    console.print(table)
    console.print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
