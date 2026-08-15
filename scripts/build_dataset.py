"""
One-off dataset build script - runs the WebCrawler against a curated set of
Wikipedia seed pages (grouped into 6 categories) to populate the bundled,
offline-usable corpus at IR/Assignment 2/data/. This demonstrates genuine
web crawling (Section B) while giving the Streamlit app a reliable dataset
that also works without a live network connection during grading.

Run manually:  python scripts/build_dataset.py
The Streamlit crawling interface can also crawl additional/live seed URLs
at run time with a configurable depth - this script just seeds the initial
corpus so every other module (index, search, ranking, recommender, eval)
has data to work with out of the box.
"""

import json
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_engine.crawler import WebCrawler, _normalize_url

SEEDS = {
    "Technology": [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Cloud_computing",
        "https://en.wikipedia.org/wiki/Computer_security",
        "https://en.wikipedia.org/wiki/Internet_of_things",
        "https://en.wikipedia.org/wiki/Quantum_computing",
        "https://en.wikipedia.org/wiki/5G",
        "https://en.wikipedia.org/wiki/Blockchain",
        "https://en.wikipedia.org/wiki/Robotics",
        "https://en.wikipedia.org/wiki/Big_data",
    ],
    "Business": [
        "https://en.wikipedia.org/wiki/Stock_market",
        "https://en.wikipedia.org/wiki/Startup_company",
        "https://en.wikipedia.org/wiki/Supply_chain",
        "https://en.wikipedia.org/wiki/E-commerce",
        "https://en.wikipedia.org/wiki/Inflation",
        "https://en.wikipedia.org/wiki/Venture_capital",
        "https://en.wikipedia.org/wiki/Cryptocurrency",
        "https://en.wikipedia.org/wiki/Mergers_and_acquisitions",
        "https://en.wikipedia.org/wiki/Globalization",
    ],
    "Sports": [
        "https://en.wikipedia.org/wiki/Association_football",
        "https://en.wikipedia.org/wiki/Basketball",
        "https://en.wikipedia.org/wiki/Olympic_Games",
        "https://en.wikipedia.org/wiki/Cricket",
        "https://en.wikipedia.org/wiki/Tennis",
        "https://en.wikipedia.org/wiki/Formula_One",
        "https://en.wikipedia.org/wiki/Marathon",
        "https://en.wikipedia.org/wiki/Swimming_(sport)",
        "https://en.wikipedia.org/wiki/Rugby_football",
    ],
    "Health": [
        "https://en.wikipedia.org/wiki/Nutrition",
        "https://en.wikipedia.org/wiki/Mental_health",
        "https://en.wikipedia.org/wiki/Vaccine",
        "https://en.wikipedia.org/wiki/Obesity",
        "https://en.wikipedia.org/wiki/Cardiovascular_disease",
        "https://en.wikipedia.org/wiki/Sleep",
        "https://en.wikipedia.org/wiki/Physical_exercise",
        "https://en.wikipedia.org/wiki/Public_health",
        "https://en.wikipedia.org/wiki/Diabetes",
    ],
    "Science": [
        "https://en.wikipedia.org/wiki/Climate_change",
        "https://en.wikipedia.org/wiki/Genetics",
        "https://en.wikipedia.org/wiki/Space_exploration",
        "https://en.wikipedia.org/wiki/Renewable_energy",
        "https://en.wikipedia.org/wiki/Evolution",
        "https://en.wikipedia.org/wiki/Astronomy",
        "https://en.wikipedia.org/wiki/Neuroscience",
        "https://en.wikipedia.org/wiki/Particle_physics",
        "https://en.wikipedia.org/wiki/Biotechnology",
    ],
    "Entertainment": [
        "https://en.wikipedia.org/wiki/Film_industry",
        "https://en.wikipedia.org/wiki/Streaming_media",
        "https://en.wikipedia.org/wiki/Music_industry",
        "https://en.wikipedia.org/wiki/Video_game_industry",
        "https://en.wikipedia.org/wiki/Television",
        "https://en.wikipedia.org/wiki/Animation",
        "https://en.wikipedia.org/wiki/Social_media",
        "https://en.wikipedia.org/wiki/Podcast",
        "https://en.wikipedia.org/wiki/Streaming_television",
    ],
}


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)

    all_records = []
    all_edges = []
    doc_counter = 0
    seen_titles = set()

    for category, urls in SEEDS.items():
        crawler = WebCrawler(max_depth=0, max_pages=len(urls), timeout=10, delay=0.1)
        docs, metadata, edges = crawler.crawl(urls)
        print(f"[{category}] fetched={len(docs)} stats={crawler.stats}")

        local_id_map = {}
        for doc_id, doc in docs.items():
            meta = metadata[doc_id]
            if doc["title"] in seen_titles:
                continue
            seen_titles.add(doc["title"])
            global_id = f"doc_{doc_counter:04d}"
            doc_counter += 1
            local_id_map[doc_id] = global_id
            all_records.append({
                "doc_id": global_id,
                "title": doc["title"],
                "url": meta["url"],
                "category": category,
                "content": doc["content"],
                "crawl_depth": meta["crawl_depth"],
                "discovered_from": meta["discovered_from"],
                "timestamp": meta["timestamp"],
                "source": "wikipedia_crawl",
            })

    # Discover the real hyperlink graph among the curated seed pages: fetch
    # each page once more just for its outlink list, then keep only edges
    # that land on another page inside our fixed 54-document corpus.
    url_to_doc_id = {_normalize_url(r["url"]): r["doc_id"] for r in all_records}
    link_crawler = WebCrawler(max_depth=0, timeout=10, delay=0.1)
    for record in all_records:
        _, _, out_links = link_crawler.fetch_single(record["url"])
        for link in out_links:
            target_doc = url_to_doc_id.get(link)
            if target_doc and target_doc != record["doc_id"]:
                all_edges.append([record["doc_id"], target_doc])
    all_edges = [list(e) for e in {tuple(e) for e in all_edges}]

    articles_path = os.path.join(data_dir, "articles.json")
    with open(articles_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    edges_path = os.path.join(data_dir, "link_graph.json")
    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump(all_edges, f, indent=2)

    print(f"\nSaved {len(all_records)} articles to {articles_path}")
    print(f"Saved {len(all_edges)} link-graph edges to {edges_path}")

    categories = {}
    for r in all_records:
        categories.setdefault(r["category"], 0)
        categories[r["category"]] += 1
    print("Category distribution:", categories)


if __name__ == "__main__":
    main()
