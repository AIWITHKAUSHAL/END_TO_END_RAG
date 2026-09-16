# Evaluation Notes

A RAG application should be evaluated in two parts: retrieval and generation. Retrieval evaluation checks whether the top-k chunks contain the evidence needed to answer the question. Generation evaluation checks whether the final answer is faithful to the retrieved context.

Common retrieval issues include chunks that are too small, chunks that are too large, poor OCR quality, weak metadata, and mismatched embedding models. Common generation issues include missing citations, using outside knowledge, and answering even when the context is insufficient.

For a classroom demonstration, a good walkthrough shows the raw documents, cleaned text, chunks, embeddings, FAISS index files, query embedding, similarity scores, retrieved chunks, final prompt, generated answer, and returned sources.
