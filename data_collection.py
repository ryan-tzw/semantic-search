import arxiv
import json


def fetch_papers(categories, max_results: int = 100):
    """
    Fetch the latest papers from multiple arXiv categories.
    """
    client = arxiv.Client()

    # Build a Boolean OR query across specified categories
    query = " OR ".join(f"cat:{c}" for c in categories)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        papers.append(
            {
                "id": result.entry_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "published": result.published.isoformat(),
                "categories": result.categories,
            }
        )
    return papers


if __name__ == "__main__":
    # Define target categories and fetch metadata
    cats = ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]
    data = fetch_papers(cats, max_results=200)

    # Persist metadata to JSON
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fetched metadata for {len(data)} papers from categories: {cats}")
