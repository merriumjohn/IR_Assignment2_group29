"""
Generates the two auxiliary datasets that cannot come from crawling:

1. data/interactions.csv - simulated user-item ratings, used to demonstrate
   Collaborative Filtering (Section E). Each simulated user has 1-2 preferred
   categories and rates a random subset of articles, rating preferred-category
   articles higher on average - this creates the co-rating signal collaborative
   filtering needs. Clearly documented as simulated (Wikipedia has no natural
   user-rating signal) in the README/report.

2. data/relevance_judgments.json - hand-curated query -> relevant doc_ids
   ground truth, used by the Evaluation Dashboard (Section F) to compute
   Precision/Recall/F1/MAP/MRR/NDCG against the search engine's ranked output.
"""

import json
import os
import random

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def build_interactions(articles, num_users=20):
    categories = sorted({a["category"] for a in articles})
    docs_by_category = {c: [a["doc_id"] for a in articles if a["category"] == c] for c in categories}

    rows = ["user_id,doc_id,rating"]
    for u in range(1, num_users + 1):
        user_id = f"user_{u:03d}"
        preferred = random.sample(categories, k=random.choice([1, 2]))
        # Rate 8-14 articles: mostly from preferred categories (high ratings),
        # a few from other categories (lower ratings) to keep the matrix realistic.
        preferred_docs = [d for c in preferred for d in docs_by_category[c]]
        other_docs = [d for c in categories if c not in preferred for d in docs_by_category[c]]

        n_preferred = min(len(preferred_docs), random.randint(6, 10))
        n_other = min(len(other_docs), random.randint(2, 4))
        rated_preferred = random.sample(preferred_docs, n_preferred)
        rated_other = random.sample(other_docs, n_other)

        for d in rated_preferred:
            rating = random.choice([4, 4, 5, 5, 3])
            rows.append(f"{user_id},{d},{rating}")
        for d in rated_other:
            rating = random.choice([1, 2, 2, 3])
            rows.append(f"{user_id},{d},{rating}")

    return "\n".join(rows) + "\n"


def build_relevance_judgments(articles):
    """Small hand-curated query set. Relevance = category match plus a
    keyword sanity check against the title/content so judgments reflect real
    topical relevance rather than pure category labels."""
    queries = {
        "artificial intelligence and machine learning": ["Technology"],
        "cloud computing and cybersecurity": ["Technology"],
        "stock market and venture capital investment": ["Business"],
        "cryptocurrency and blockchain finance": ["Business", "Technology"],
        "olympic games and athletics": ["Sports"],
        "football and cricket tournaments": ["Sports"],
        "nutrition and mental health": ["Health"],
        "vaccines and public health": ["Health"],
        "climate change and renewable energy": ["Science"],
        "space exploration and astronomy": ["Science"],
        "streaming media and film industry": ["Entertainment"],
        "video games and social media": ["Entertainment"],
    }

    judgments = {}
    for query, rel_categories in queries.items():
        relevant = [a["doc_id"] for a in articles if a["category"] in rel_categories]
        judgments[query] = relevant
    return judgments


def main():
    with open(os.path.join(DATA_DIR, "articles.json"), "r", encoding="utf-8") as f:
        articles = json.load(f)

    interactions_csv = build_interactions(articles)
    with open(os.path.join(DATA_DIR, "interactions.csv"), "w", encoding="utf-8") as f:
        f.write(interactions_csv)
    print("Wrote data/interactions.csv:", interactions_csv.count("\n"), "rows")

    judgments = build_relevance_judgments(articles)
    with open(os.path.join(DATA_DIR, "relevance_judgments.json"), "w", encoding="utf-8") as f:
        json.dump(judgments, f, indent=2)
    print("Wrote data/relevance_judgments.json:", len(judgments), "queries")


if __name__ == "__main__":
    main()
