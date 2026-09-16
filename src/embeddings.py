from google import genai


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return [embedding.values for embedding in response.embeddings]
