import json


def load_metadata(path: str):
    """Load the arXiv metadata JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_abstracts(metadata: list):
    """
    Convert metadata into single-chunk abstracts with titles.
    """
    return [
        {
            "paper_id": paper["id"],
            "title": paper.get("title", ""),
            "chunk_index": 0,
            "text": paper.get("abstract", ""),
        }
        for paper in metadata
    ]


if __name__ == "__main__":
    papers = load_metadata("metadata.json")
    paper_chunks = chunk_abstracts(papers)
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(paper_chunks, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(paper_chunks)} single-chunk abstracts (with titles).")
