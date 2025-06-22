import json
from sentence_transformers import SentenceTransformer


def load_chunks(path: str):
    """Load pre-chunked chunks with titles."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_locally(chunks, model_name: str = "all-MiniLM-L6-v2"):
    """
    Generate embeddings locally using SentenceTransformers.
    Returns a list of dicts with metadata + embedding.
    """
    # Initialize the model
    model = SentenceTransformer(model_name)
    embeddings = []
    texts = [c["text"] for c in chunks]

    # Batch encode texts
    vectors = model.encode(texts, show_progress_bar=True)

    for chunk, vec in zip(chunks, vectors):
        embeddings.append(
            {
                "paper_id": chunk.get("paper_id"),
                "title": chunk.get("title", ""),
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk.get("text"),
                "embedding": vec.tolist(),
            }
        )
    return embeddings


if __name__ == "__main__":
    chunks = load_chunks("chunks.json")
    embeddings = embed_locally(chunks, model_name="all-MiniLM-L6-v2")

    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    print(f"✓ Generated embeddings for {len(embeddings)} chunks (with titles).")
