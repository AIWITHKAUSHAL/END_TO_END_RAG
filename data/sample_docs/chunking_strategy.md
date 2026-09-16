# Chunking Strategy

This project compares three chunk sizes: 250 words, 500 words, and 900 words. The comparison demonstrates how retrieval changes when the indexed units become smaller or larger.

A 250 word chunk is useful for short factual questions because the retrieved context is narrow. It often gives higher precision, but a generated answer may need several chunks to recover the full story.

A 500 word chunk is a balanced default for class assignments. It keeps enough context for the language model while avoiding very broad retrieval results.

A 900 word chunk is useful when documents are narrative or when the answer depends on long passages. The downside is that similarity search may return a chunk because part of it is relevant even though much of the chunk is not.

The best chunk size depends on the document type, the question style, the embedding model, the top-k value, and the amount of context that the generation model can accept. A good RAG system measures retrieval quality instead of guessing.
