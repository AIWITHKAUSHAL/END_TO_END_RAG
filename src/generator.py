"""Build grounded RAG prompts and generate answers with Gemini."""

from google import genai


SYSTEM_INSTRUCTION = """You are a careful RAG assistant.
Answer only from the provided context.
If the context is not enough, say what is missing.
Cite sources using [source: chunk N] after relevant claims."""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Combine retrieved text and provenance with the user's question."""

    context_blocks = []
    for item in retrieved_chunks:
        chunk = item["chunk"]
        context_blocks.append(
            f"Source: {chunk.source}\n"
            f"Chunk: {chunk.chunk_index}\n"
            f"Similarity: {item['score']:.4f}\n"
            f"Text:\n{chunk.text}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"{SYSTEM_INSTRUCTION}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"


class GeminiGenerator:
    """Validated Gemini client for grounded answer generation."""

    def __init__(self, api_key: str, model: str):
        """Create a live client after rejecting missing keys and mock models."""

        if not api_key.strip():
            raise ValueError("A Google API key is required for live Gemini requests.")
        if any(marker in model.lower() for marker in ("demo", "mock", "fake", "test")):
            raise ValueError(f"Refusing non-production generation model: {model}")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate and return a non-empty text response for a prompt."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        if not response.text:
            raise RuntimeError("Gemini returned no text response.")
        return response.text

    def verify_connection(self) -> str:
        """Make a minimal live request and return the model's response."""
        return self.generate("Reply with exactly: LIVE")
