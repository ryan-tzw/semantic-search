# ArXiv Semantic Search Retriever

A lightweight **semantic search engine** for arXiv abstracts, built for seamless integration into a larger Retrieval-Augmented Generation (RAG) pipeline.

## Project Overview

This repository implements a semantic search engine over arXiv abstracts. It handles:

-   Fetching metadata across multiple categories
-   Preprocessing each paper into single chunks
-   Generating embeddings locally with SentenceTransformers
-   Storing and indexing vectors in a persistent ChromaDB collection
-   Running an interactive CLI that returns the most similar abstracts for any user query

## Key Features

-   **Multi-Category Fetch**: Boolean-OR queries against cs.AI, cs.CL, cs.CV, cs.LG
-   **Abstract-Only Chunking**: One coherent chunk per paper with metadata
-   **Local Embeddings**: `all-MiniLM-L6-v2` for 384-dim vectors, no online API calls
-   **Persistent Vector Store**: ChromaDB-backed nearest-neighbor search

## RAG Context

Retrieval-Augmented Generation (RAG) enhances LLM outputs by **retrieving** relevant external documents and then **generating** responses grounded in those sources. This repository covers the **indexing** and **retrieval** stages of RAG:

1. **Indexing**: Turning abstracts into embeddings and storing them in a vector store.
2. **Retrieval**: Embedding user queries and fetching the top-k most similar abstracts.

Integration with an LLM for the **augmentation** and **generation** steps completes the full RAG pipeline, but that is not done in this project.

## Architecture & Pipeline Flow

1. **Data Collection**: `data_collection.py` builds and runs a multi-category arXiv query → outputs JSON metadata
2. **Preprocessing**: `preprocess.py` wraps each result into a single chunk record
3. **Embeddings**: `embed.py` batch-encodes text chunks locally → outputs JSON of vectors
4. **Vector Store**: `vectorstore_setup.py` ingests vectors into a persistent ChromaDB collection
5. **Retrieval**: `retrieve.py` runs a CLI loop—embed query, search, display title + truncated snippet

## Installation & Prerequisites

-   **Python 3.7+**
-   (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
    ```

*   **Install dependencies**:

    ```bash
    pip install -r requirements.txt

    OR

    conda env create -f environment.yml -p ./.conda
    ```

## Quickstart Guide

1. **Run the full pipeline** (fetch → preprocess → embed → ingest):

    ```bash
    python run_pipeline.py
    ```

2. **Launch semantic search REPL**:

    ```bash
    python retrieve.py
    ```

    Type any natural-language query and see the top-K matching abstracts with titles.

3. **Exit**: type `exit` or `quit`.

## Configuration

-   **Categories**: edit the list in `data_collection.py` to include any arXiv categories you need.
-   **Top‐K results & snippet length**: modify the constants in `retrieve.py`.

## Future Improvements

-   **Web UI**: wrap `retrieve.py` in Streamlit/Gradio for a browser interface.
-   **Full RAG**: after retrieval, build a prompt with the snippets and dispatch to an LLM for generation.
-   **PDF support**: extend `data_collection.py` to download PDFs and feed into a PDF-to-text extractor.
-   **Feedback loop**: log user selections to re-rank or fine-tune embeddings over time.
