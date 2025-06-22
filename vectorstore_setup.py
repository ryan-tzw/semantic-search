import json
from chromadb import PersistentClient


def load_embeddings(path: str):
    """Load the embeddings (with title) from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_embeddings("embeddings.json")

    # Prep parallel lists for ingestion
    ids = [f"{item['paper_id']}_{item['chunk_index']}" for item in data]
    embeddings = [item["embedding"] for item in data]
    metadatas = [
        {
            "paper_id": item["paper_id"],
            "title": item.get("title", ""),
            "chunk_index": item["chunk_index"],
        }
        for item in data
    ]
    documents = [item["text"] for item in data]

    client = PersistentClient(path="./chroma_db_store")
    collection = client.get_or_create_collection(name="arxiv")
    collection.add(
        ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
    )

    print(f"Ingested {len(ids)} vectors (with titles) into 'arxiv'.")
