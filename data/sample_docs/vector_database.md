# Vector Database Notes

FAISS is a vector similarity search library. In this assignment it acts as the vector database. The application stores normalized embedding vectors in a FAISS index and keeps chunk metadata in a JSON file beside the index.

Cosine similarity can be implemented in FAISS by normalizing vectors and then using inner product search. Higher similarity scores mean the query vector and chunk vector point in a more similar semantic direction.

The source metadata is just as important as the vector. Without source metadata, the application could answer a question but would not be able to show which document and chunk supported the answer. A useful RAG answer should include source names, chunk numbers, and the text used by the LLM.
