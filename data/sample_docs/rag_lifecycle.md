# RAG Lifecycle Notes

Retrieval Augmented Generation, usually called RAG, connects a language model to an external knowledge base. The goal is to reduce hallucination, keep answers grounded, and make private or recent documents available at question-answering time.

The ingestion side starts with documents. Documents may be PDFs, Markdown notes, text files, web pages, transcripts, or support tickets. A parser extracts raw text from each file. A cleaner removes noise such as repeated blank lines, broken spacing, headers, footers, and unreadable characters.

After cleaning, the text is split into chunks. Chunking is important because embedding models and vector databases work best when each stored unit is focused. Small chunks can retrieve precise details, but they may lose surrounding context. Large chunks preserve context, but they may retrieve extra unrelated information. Overlap helps preserve meaning across chunk boundaries.

Each chunk is converted into an embedding. An embedding is a numeric vector that captures semantic meaning. Similar ideas should have vectors that are close together. The vectors and chunk metadata are stored in a vector database such as FAISS, Chroma, Qdrant, Pinecone, or Weaviate.

At query time, the user question is embedded with the same embedding model. The vector database performs similarity search and returns the top-k most relevant chunks. These chunks are inserted into a prompt with the question. The LLM uses the retrieved context to write an answer and cite the sources used.
