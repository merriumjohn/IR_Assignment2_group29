"""
Configurable web crawler.

Requirements covered (Section B of the assignment):
- Acquire information via web crawling from one or more heterogeneous seed sources.
- Support configurable crawling depth and multiple seed sources.
- Handle duplicate URLs and duplicate documents (near-duplicate content hashing).
- Store extracted metadata separately from document contents.
"""

import hashlib
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_AGENT = "IR-Assignment2-Bot/1.0 (BITS WILP; educational use)"

# Wikipedia namespaces / link patterns to skip while following outlinks
SKIP_LINK_MARKERS = (
    "Special:", "Help:", "Talk:", "User:", "Wikipedia:", "File:", "Template:",
    "Category:", "Portal:", "Draft:", "TimedText:", "Module:", "MediaWiki:",
    "#", "action=", "index.php",
)


def _normalize_url(url: str) -> str:
    """Strip fragments/query noise so the same page isn't re-queued twice."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def _content_hash(text: str) -> str:
    """Hash of normalized text used to detect duplicate/near-duplicate documents."""
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


class WebCrawler:
    """Breadth-first crawler with depth limit, page cap and dedup."""

    def __init__(self, max_depth=1, max_pages=50, per_page_link_limit=5,
                 timeout=8, delay=0.15, allowed_domains=None):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.per_page_link_limit = per_page_link_limit
        self.timeout = timeout
        self.delay = delay
        self.allowed_domains = allowed_domains  # None = same domain as each seed

        # Populated after crawl()
        self.documents = {}   # doc_id -> {"title":..., "content":...}
        self.metadata = {}    # doc_id -> {url, depth, discovered_from, timestamp, content_hash, out_links}
        self.edges = []       # list of (src_doc_id, dst_doc_id) for link-graph based ranking
        self.stats = {"fetched": 0, "duplicate_urls": 0, "duplicate_content": 0, "errors": 0}

    def _fetch(self, url):
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=self.timeout, verify=False)
        resp.raise_for_status()
        return resp.text

    def _extract(self, html, base_url):
        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find(id="firstHeading") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else base_url

        content_root = soup.find(id="mw-content-text") or soup.find("body") or soup
        paragraphs = [p.get_text(" ", strip=True) for p in content_root.find_all("p")]
        text = "\n".join(p for p in paragraphs if p)

        links = []
        for a in content_root.find_all("a", href=True):
            href = a["href"]
            if any(marker in href for marker in SKIP_LINK_MARKERS):
                continue
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.scheme not in ("http", "https"):
                continue
            links.append(_normalize_url(full))

        return title, text, links

    def fetch_single(self, url):
        """Fetch + extract a single page without queuing/BFS bookkeeping.
        Returns (title, text, out_links) or (None, None, []) on failure.
        Used to discover the hyperlink graph among a fixed, curated set of
        seed pages without letting the crawl grow beyond that set.
        """
        try:
            html = self._fetch(url)
        except Exception:
            return None, None, []
        finally:
            time.sleep(self.delay)
        return self._extract(html, url)

    def crawl(self, seed_urls):
        """Run BFS crawl starting from one or more seed URLs.

        Returns (documents, metadata, edges) - documents/metadata are keyed by
        an internal doc_id so content and metadata are stored as separate
        structures (never merged into one record).
        """
        allowed = set(self.allowed_domains) if self.allowed_domains else None
        if allowed is None:
            allowed = {urlparse(u).netloc for u in seed_urls}

        seen_urls = set()
        seen_content_hashes = set()
        url_to_doc_id = {}
        queue = deque()

        for seed in seed_urls:
            norm = _normalize_url(seed)
            queue.append((norm, 0, None))
            seen_urls.add(norm)

        doc_counter = 0
        while queue and doc_counter < self.max_pages:
            url, depth, parent = queue.popleft()
            try:
                html = self._fetch(url)
            except Exception:
                self.stats["errors"] += 1
                continue
            finally:
                time.sleep(self.delay)

            title, text, out_links = self._extract(html, url)
            if not text or len(text.split()) < 20:
                self.stats["errors"] += 1
                continue

            chash = _content_hash(text)
            if chash in seen_content_hashes:
                self.stats["duplicate_content"] += 1
                continue
            seen_content_hashes.add(chash)

            doc_id = f"doc_{doc_counter:04d}"
            doc_counter += 1
            url_to_doc_id[url] = doc_id

            self.documents[doc_id] = {"title": title, "content": text}
            self.metadata[doc_id] = {
                "url": url,
                "crawl_depth": depth,
                "discovered_from": parent,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "content_hash": chash,
                "out_link_count": len(out_links),
            }
            self.stats["fetched"] += 1

            if parent and parent in url_to_doc_id:
                self.edges.append((url_to_doc_id[parent], doc_id))

            if depth < self.max_depth:
                for link in out_links[: self.per_page_link_limit]:
                    if urlparse(link).netloc not in allowed:
                        continue
                    if link in seen_urls:
                        self.stats["duplicate_urls"] += 1
                        continue
                    seen_urls.add(link)
                    queue.append((link, depth + 1, url))

        # second pass: resolve edges that point to pages fetched later (BFS order)
        resolved_edges = []
        for src_url, dst_doc in [(m["discovered_from"], d) for d, m in self.metadata.items()]:
            if src_url in url_to_doc_id:
                resolved_edges.append((url_to_doc_id[src_url], dst_doc))
        self.edges = list(set(resolved_edges))

        return self.documents, self.metadata, self.edges


def load_local_dataset(articles_path):
    """Fallback loader for the bundled pre-crawled dataset (offline/no-network use).

    Returns (documents, metadata) in the same shape produced by WebCrawler.crawl,
    so the rest of the pipeline is agnostic to the data source.
    """
    import json

    with open(articles_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    documents, metadata = {}, {}
    for rec in records:
        doc_id = rec["doc_id"]
        documents[doc_id] = {"title": rec["title"], "content": rec["content"]}
        metadata[doc_id] = {
            "url": rec.get("url", ""),
            "category": rec.get("category", "Uncategorized"),
            "crawl_depth": rec.get("crawl_depth", 0),
            "discovered_from": rec.get("discovered_from"),
            "timestamp": rec.get("timestamp", ""),
            "content_hash": _content_hash(rec["content"]),
            "source": rec.get("source", "bundled_dataset"),
        }
    return documents, metadata
