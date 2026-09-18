"""Create document and query vector embeddings with the Gemini API."""

from google import genai


class GeminiEmbedder:
    """Small adapter around Gemini's batch content-embedding endpoint."""

    def __init__(self, api_key: str, model: str):
        """Create an API client that uses the selected embedding model."""

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one numeric embedding per input string, or an empty list."""

        if not texts:
            return []

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return [embedding.values for embedding in response.embeddings]
