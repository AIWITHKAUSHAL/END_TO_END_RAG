# Video Walkthrough Script

1. Show the project folder and explain the assignment architecture: Documents -> Parsing -> Cleaning -> Chunking -> Embeddings -> Vector Database -> User Query -> Query Embedding -> Similarity Search -> Top-K Chunks -> Prompt -> LLM -> Answer + Sources.
2. Open `data/sample_docs` and show the raw Markdown documents.
3. Open `src/document_loader.py`, `src/cleaner.py`, and `src/chunker.py` to explain parsing, cleaning, and word-based chunking with overlap.
4. Open `src/embeddings.py` and explain that Gemini creates vectors for every chunk using `gemini-embedding-001`.
5. Open `src/vector_store.py` and explain that FAISS stores normalized vectors and JSON stores source metadata.
6. Run `python ingest.py --chunk-size 500` and show the generated files in `indexes`.
7. Run `python query.py "What are the main stages in a RAG pipeline?" --chunk-size 500` and show answer plus source chunks.
8. Run `python compare_chunks.py` and explain the retrieval differences for 250, 500, and 900 word chunks.
9. Run `streamlit run app.py` and demonstrate the same lifecycle in the UI.
10. Close by explaining why the answer is grounded: the prompt contains only top-k retrieved chunks, and the UI/CLI returns the exact sources used.
