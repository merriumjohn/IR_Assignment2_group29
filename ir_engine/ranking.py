"""
Web searching & ranking (Section D).

- VectorSpaceSearch: TF-IDF cosine-similarity ranked retrieval with a simple
  query-optimization step (stopword removal + OR-query expansion via the
  inverted index postings so only candidate docs are scored).
- PageRank: classic power-iteration PageRank over a document link graph.
- HITS: hub/authority scores over the same graph.
- combined_ranking(): shows how blending a link-based signal with TF-IDF
  changes result order - used by the Ranking Visualization page.
"""

import math

import numpy as np
from scipy.sparse import diags, lil_matrix

from ir_engine.preprocessing import clean_tokenize, STOPWORDS


def _filtered_tokens(text):
    """Tokenize and drop stopwords."""
    return [t for t in clean_tokenize(text) if t not in STOPWORDS]


class VectorSpaceSearch:
    """Custom TF-IDF vector-space search built on an in-memory sparse matrix.

    No reliance on scikit-learn for the scoring pipeline — we compute raw TF,
    smoothed IDF, and L2-normalized TF-IDF vectors directly from the inverted
    index logic.
    """

    def __init__(self, doc_ids, texts):
        self.doc_ids = doc_ids
        n = len(doc_ids)

        # Build vocabulary and per-document term counts.
        self.vocab = {}          # term -> column index
        doc_counts = []          # list of {col: tf}
        for text in texts:
            counts = {}
            for t in _filtered_tokens(text):
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                col = self.vocab[t]
                counts[col] = counts.get(col, 0) + 1
            doc_counts.append(counts)

        m = len(self.vocab)

        # Smoothed IDF: log((1 + N) / (1 + df)) + 1, matching common TF-IDF.
        self.idf = np.zeros(m, dtype=np.float64)
        for t, col in self.vocab.items():
            df = sum(1 for counts in doc_counts if col in counts)
            self.idf[col] = math.log((1 + n) / (1 + df)) + 1.0

        # Build sparse TF-IDF document matrix and L2-normalize rows.
        mat = lil_matrix((n, m), dtype=np.float64)
        for i, counts in enumerate(doc_counts):
            for col, tf in counts.items():
                mat[i, col] = tf * self.idf[col]
        mat = mat.tocsr()
        norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        self.doc_matrix = diags(1.0 / norms).dot(mat).tocsr()

        # Fast lookup for which docs contain a term (for candidate pruning).
        self.postings = {
            t: {i for i, counts in enumerate(doc_counts) if c in counts}
            for t, c in self.vocab.items()
        }

    def search(self, query, top_k=10, candidate_doc_ids=None):
        q_counts = {}
        for t in _filtered_tokens(query):
            if t in self.vocab:
                col = self.vocab[t]
                q_counts[col] = q_counts.get(col, 0) + 1

        if not q_counts:
            return []

        m = len(self.vocab)
        q_vec = lil_matrix((1, m), dtype=np.float64)
        for col, tf in q_counts.items():
            q_vec[0, col] = tf * self.idf[col]
        q_vec = q_vec.tocsr()

        q_norm = float(np.sqrt(q_vec.multiply(q_vec).sum()))
        if q_norm > 0:
            q_vec = q_vec * (1.0 / q_norm)

        # Cosine similarity = dot product because both vectors are L2-normalized.
        sims = q_vec.dot(self.doc_matrix.T).toarray().flatten()

        if candidate_doc_ids is not None:
            mask = np.array([1.0 if d in candidate_doc_ids else 0.0 for d in self.doc_ids])
            sims = sims * mask

        ranked_idx = np.argsort(-sims)
        results = [(self.doc_ids[i], float(sims[i])) for i in ranked_idx if sims[i] > 0]
        return results[:top_k]

    def candidate_docs_from_index(self, query, inverted_index):
        """OR-query optimization: only score docs containing >=1 query term."""
        terms = [t for t in clean_tokenize(query) if t not in STOPWORDS]
        candidates = set()
        for t in terms:
            candidates.update(inverted_index.postings.get(t, {}).keys())
        return candidates or None


def page_rank(doc_ids, edges, damping=0.85, max_iter=100, tol=1e-6):
    """Power-iteration PageRank. Falls back to a uniform similarity graph
    if `edges` is empty (e.g. offline/local dataset with no hyperlink graph)."""
    n = len(doc_ids)
    if n == 0:
        return {}
    idx = {d: i for i, d in enumerate(doc_ids)}
    out_links = [[] for _ in range(n)]
    in_links = [[] for _ in range(n)]

    for src, dst in edges:
        if src in idx and dst in idx and src != dst:
            out_links[idx[src]].append(idx[dst])
            in_links[idx[dst]].append(idx[src])

    scores = np.full(n, 1.0 / n)
    out_degree = np.array([len(o) for o in out_links])

    for _ in range(max_iter):
        new_scores = np.full(n, (1 - damping) / n)
        dangling_mass = damping * scores[out_degree == 0].sum() / n
        for i in range(n):
            for j in in_links[i]:
                new_scores[i] += damping * scores[j] / out_degree[j]
        new_scores += dangling_mass
        if np.abs(new_scores - scores).sum() < tol:
            scores = new_scores
            break
        scores = new_scores

    return {doc_ids[i]: float(scores[i]) for i in range(n)}


def hits(doc_ids, edges, max_iter=100, tol=1e-6):
    """HITS algorithm returning (hub_scores, authority_scores)."""
    n = len(doc_ids)
    if n == 0:
        return {}, {}
    idx = {d: i for i, d in enumerate(doc_ids)}
    out_links = [[] for _ in range(n)]
    in_links = [[] for _ in range(n)]
    for src, dst in edges:
        if src in idx and dst in idx and src != dst:
            out_links[idx[src]].append(idx[dst])
            in_links[idx[dst]].append(idx[src])

    hub = np.full(n, 1.0)
    auth = np.full(n, 1.0)

    for _ in range(max_iter):
        new_auth = np.array([sum(hub[j] for j in in_links[i]) for i in range(n)])
        norm = np.linalg.norm(new_auth) or 1.0
        new_auth = new_auth / norm

        new_hub = np.array([sum(new_auth[j] for j in out_links[i]) for i in range(n)])
        norm = np.linalg.norm(new_hub) or 1.0
        new_hub = new_hub / norm

        if np.abs(new_auth - auth).sum() + np.abs(new_hub - hub).sum() < tol:
            auth, hub = new_auth, new_hub
            break
        auth, hub = new_auth, new_hub

    return ({doc_ids[i]: float(hub[i]) for i in range(n)},
            {doc_ids[i]: float(auth[i]) for i in range(n)})


def combined_ranking(tfidf_results, link_scores, alpha=0.7):
    """Blend TF-IDF similarity with a normalized link-based score (PageRank/authority).

    alpha weights TF-IDF; (1-alpha) weights the link signal. Demonstrates how
    ranking changes when link importance is folded into pure content similarity.
    """
    if not tfidf_results:
        return []
    max_link = max(link_scores.values()) if link_scores else 1.0
    max_link = max_link or 1.0
    blended = []
    for doc_id, sim in tfidf_results:
        link_score = link_scores.get(doc_id, 0.0) / max_link
        blended.append((doc_id, alpha * sim + (1 - alpha) * link_score, sim, link_score))
    blended.sort(key=lambda x: x[1], reverse=True)
    return blended
