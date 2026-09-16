import argparse

from rich.console import Console
from rich.table import Table

from src.config import settings
from src.pipeline import ingest


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse, clean, chunk, embed, and index documents.")
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--chunk-size", type=int, default=settings.default_chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.default_chunk_overlap)
    args = parser.parse_args()

    result = ingest(args.data_dir, args.chunk_size, args.overlap)

    table = Table(title="Ingestion Complete")
    table.add_column("Item")
    table.add_column("Value")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":
    main()
