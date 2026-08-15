"""
Inverted index construction & management (Section D support).

- Handles duplicate documents by content-hash before indexing.
- Exposes index statistics used in the Index Management dashboard.
- Simple JSON-serializable structure so the index can be saved/loaded.
"""

import json
from collections import defaultdict


class InvertedIndex:
    def __init__(self):
        self.postings = defaultdict(dict)   # term -> {doc_id: tf}
        self.doc_lengths = {}                # doc_id -> token count
        self.doc_freq = {}                   # term -> document frequency
        self.num_docs = 0
        self._indexed_hashes = set()
        self.skipped_duplicates = []

    def build(self, tokenized_docs: dict, content_hashes: dict = None):
        """tokenized_docs: {doc_id: [tokens]}; content_hashes: {doc_id: hash} for dedup."""
        content_hashes = content_hashes or {}
        for doc_id, tokens in tokenized_docs.items():
            chash = content_hashes.get(doc_id)
            if chash and chash in self._indexed_hashes:
                self.skipped_duplicates.append(doc_id)
                continue
            if chash:
                self._indexed_hashes.add(chash)

            self.doc_lengths[doc_id] = len(tokens)
            term_counts = defaultdict(int)
            for t in tokens:
                term_counts[t] += 1
            for term, tf in term_counts.items():
                self.postings[term][doc_id] = tf

        self.num_docs = len(self.doc_lengths)
        self.doc_freq = {term: len(postings) for term, postings in self.postings.items()}
        return self

    def stats(self):
        total_postings = sum(len(p) for p in self.postings.values())
        avg_postings_len = total_postings / len(self.postings) if self.postings else 0
        return {
            "num_documents": self.num_docs,
            "num_terms": len(self.postings),
            "total_postings": total_postings,
            "avg_postings_per_term": round(avg_postings_len, 2),
            "duplicates_skipped": len(self.skipped_duplicates),
        }

    def to_json(self):
        return json.dumps({
            "postings": self.postings,
            "doc_lengths": self.doc_lengths,
            "doc_freq": self.doc_freq,
            "num_docs": self.num_docs,
        })

    @classmethod
    def from_json(cls, blob):
        data = json.loads(blob)
        idx = cls()
        idx.postings = defaultdict(dict, data["postings"])
        idx.doc_lengths = data["doc_lengths"]
        idx.doc_freq = data["doc_freq"]
        idx.num_docs = data["num_docs"]
        return idx
