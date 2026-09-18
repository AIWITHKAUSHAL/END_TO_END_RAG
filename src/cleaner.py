"""Normalize extracted document text before it is divided into chunks."""

import re


def clean_text(text: str) -> str:
    """Remove nulls, repeated whitespace, blank lines, and line-edge spaces."""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
