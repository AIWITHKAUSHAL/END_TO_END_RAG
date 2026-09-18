"""Unit tests for word-based chunking, overlap, metadata, and validation."""

import unittest

from src.chunker import chunk_text


class ChunkerTests(unittest.TestCase):
    """Verify the chunker's normal and invalid-configuration behavior."""

    def test_chunk_text_uses_overlap_and_metadata(self):
        """Chunks should repeat overlap words and preserve source settings."""

        text = " ".join(f"word{i}" for i in range(12))

        chunks = chunk_text(text, source="demo.txt", chunk_size=5, overlap=2)

        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].text, "word0 word1 word2 word3 word4")
        self.assertEqual(chunks[1].text, "word3 word4 word5 word6 word7")
        self.assertEqual(chunks[0].source, "demo.txt")
        self.assertEqual(chunks[0].chunk_size, 5)
        self.assertEqual(chunks[0].overlap, 2)

    def test_chunk_text_rejects_overlap_larger_than_chunk(self):
        """Overlap equal to the chunk size should raise a validation error."""

        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_text("hello world", source="demo.txt", chunk_size=5, overlap=5)


if __name__ == "__main__":
    unittest.main()
