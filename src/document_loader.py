from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path}")


def iter_documents(data_dir: str | Path) -> Iterable[dict]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield {
                "source": str(path.as_posix()),
                "text": load_document(path),
            }
