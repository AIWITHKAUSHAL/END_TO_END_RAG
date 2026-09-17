# End-to-End RAG Application

This project implements the complete Retrieval Augmented Generation pipeline required for the assignment:

**Documents -> Parsing -> Cleaning -> Chunking -> Embeddings -> Vector Database -> User Query -> Query Embedding -> Similarity Search -> Top-K Chunks -> Prompt -> LLM -> Answer + Sources**

It uses:

- **Gemini Embeddings**: `gemini-embedding-001`
- **Generator LLM**: `gemini-3.5-flash-lite` by default
- **Vector database**: FAISS
- **UI**: Streamlit
- **Chunk-size experiment**: 250, 500, and 900 words

## Architecture Picture

Open the diagram here:

- [docs/architecture_diagram.svg](docs/architecture_diagram.svg)
- [docs/architecture.md](docs/architecture.md)

## Project Structure

```text
.
├── app.py                         # Streamlit app
├── ingest.py                      # Builds FAISS index from documents
├── query.py                       # CLI question-answering demo
├── compare_chunks.py              # Compares retrieval for 3 chunk sizes
├── data/sample_docs/              # Sample documents for live demo
├── docs/architecture.md           # Mermaid architecture diagram
├── docs/architecture_diagram.svg  # Picture for reports/slides
├── docs/video_script.md           # YouTube walkthrough script
├── src/
│   ├── document_loader.py         # Parsing
│   ├── cleaner.py                 # Cleaning
│   ├── chunker.py                 # Chunking
│   ├── embeddings.py              # Gemini embeddings
│   ├── vector_store.py            # FAISS vector database
│   ├── generator.py               # Prompt + Gemini answer generation
│   └── pipeline.py                # End-to-end orchestration
└── requirements.txt
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```bash
copy .env.example .env
```

Add your Google API key:

```env
GOOGLE_API_KEY=
GENERATION_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
```

Verify that the configured generator makes a real Gemini API request:

```bash
python verify_llm.py
```

The application has no demo or mock fallback. It stops with an error if the API
key is missing, the configured model looks like a demo/mock model, or Gemini
does not return a text response.

## Run Ingestion

The sample folder includes short teaching notes plus the original open-access RAG paper:

```text
data/sample_docs/rag_original_paper.pdf
```

This command parses documents, cleans text, creates chunks, embeds chunks, and saves a FAISS index:

```bash
python ingest.py --data-dir data/sample_docs --chunk-size 500 --overlap 80
```

The generated files are saved in `indexes/`.

## Ask a Question

```bash
python query.py "What are the main stages in a RAG pipeline?" --chunk-size 500 --top-k 4
```

Good live-demo questions:

- `What problem does retrieval-augmented generation try to solve?`
- `How does RAG combine parametric and non-parametric memory?`
- `Why is returning provenance important in a RAG system?`
- `What are the tradeoffs between small and large chunks?`

The output includes:

- The generated answer
- Source file names
- Chunk numbers
- Similarity scores

## Compare Three Chunk Sizes

The assignment requires experimenting with at least three chunk sizes. Run:

```bash
python compare_chunks.py --question "What are the main stages in a RAG pipeline?"
```

This builds and compares indexes for:

- 250-word chunks
- 500-word chunks
- 900-word chunks

It saves the comparison to:

```text
chunk_comparison_results.csv
```

## Run the Streamlit App

```bash
streamlit run app.py
```

In the sidebar:

1. Upload one or more PDF, TXT, or MD files.
2. Select a chunk size.
3. Click **Ingest uploaded documents**.
4. Ask a question.
5. Expand the source panels to inspect the exact chunks used.

## Run Tests

```bash
python -m unittest discover -s tests
```

## Why Sources Are Returned

Each chunk stores metadata:

- `source`
- `chunk_index`
- `chunk_size`
- `overlap`
- `text`

During similarity search, FAISS returns the nearest vectors. The app maps those vector positions back to chunk metadata, then displays the exact chunks used in the prompt.

## Model Recommendation

Use `gemini-3.5-flash-lite` for the classroom demo because it is fast and cost-effective. If you want stronger answer quality for final recording, switch to:

```env
GENERATION_MODEL=gemini-3.5-flash
```

The rest of the code stays the same.

## Video Submission Guide

Use [docs/video_script.md](docs/video_script.md) as the video outline. The reviewer should see:

- Raw documents
- Parsing and cleaning code
- Chunking code
- Embedding code
- FAISS vector database code
- Query embedding
- Similarity search
- Top-K chunks
- Prompt construction
- Gemini answer
- Returned sources
- Chunk-size comparison results

## GitHub Submission

Push this folder to GitHub and submit the exact folder URL:

```text
https://github.com/you/repo/tree/main/END_TO_END_RAG
```
