import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

client = PersistentClient(path="./chroma_db_store")
collection = client.get_or_create_collection(name="arxiv")

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(text: str) -> List[float]:
    """
    Embed a query string using the same SentenceTransformer model used for documents.
    Returns a list of floats representing the embedding vector.
    """
    # model.encode returns a numpy array; convert to Python list
    return model.encode(text).tolist()


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform a nearest-neighbor search in Chroma for the given query.
    Returns the top_k most similar chunks with their metadata and similarity scores.
    """
    # Embed the query
    q_emb = embed_query(query)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Format and return hits
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append(
            {
                "paper_id": meta["paper_id"],
                "title": meta.get("title", ""),
                "chunk_index": meta["chunk_index"],
                "text": doc,
                "score": dist,
            }
        )
    return hits


def main():
    print("ArXiv Semantic Search")
    print("Type your query (or 'exit' to quit):\n")
    while True:
        query = input("▶ ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        try:
            results = retrieve(query, top_k=5)
            if not results:
                print("No results found.\n")
                continue
            print("\nTop results:\n")
            for i, hit in enumerate(results, 1):
                print(f"{i}. {hit['title']} (ID: {hit['paper_id']})")
                print(f"   Score:     {hit['score']:.4f}")

                snippet = hit["text"]
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                print(f"   Snippet:   {snippet}\n")

        except Exception as e:
            print(f"Error during retrieval: {e}\n")
    print("Goodbye!")


if __name__ == "__main__":
    main()
