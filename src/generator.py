from google import genai


SYSTEM_INSTRUCTION = """You are a careful RAG assistant.
Answer only from the provided context.
If the context is not enough, say what is missing.
Cite sources using [source: chunk N] after relevant claims."""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
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
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text or ""
