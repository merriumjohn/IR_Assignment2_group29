"""
Recommender system (Section E).

- ContentBasedRecommender: TF-IDF cosine similarity between documents.
- CollaborativeRecommender: item-based collaborative filtering over a
  simulated user-item ratings matrix.
- HybridRecommender: weighted blend of the two, with Top-K + similarity scores.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, doc_ids, tfidf_matrix):
        self.doc_ids = doc_ids
        self.tfidf_matrix = tfidf_matrix
        self.sim_matrix = cosine_similarity(tfidf_matrix)
        self.index = {d: i for i, d in enumerate(doc_ids)}

    def recommend(self, doc_id, top_k=5):
        if doc_id not in self.index:
            return []
        i = self.index[doc_id]
        sims = list(enumerate(self.sim_matrix[i]))
        sims = [(self.doc_ids[j], score) for j, score in sims if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]


class CollaborativeRecommender:
    """Item-based CF: similarity between items derived from co-rating patterns."""

    def __init__(self, interactions_df: pd.DataFrame):
        self.matrix = interactions_df.pivot_table(
            index="user_id", columns="doc_id", values="rating", fill_value=0
        )
        self.doc_ids = list(self.matrix.columns)
        item_vectors = self.matrix.T.values
        self.item_sim = cosine_similarity(item_vectors)
        self.index = {d: i for i, d in enumerate(self.doc_ids)}

    def recommend_for_item(self, doc_id, top_k=5):
        if doc_id not in self.index:
            return []
        i = self.index[doc_id]
        sims = [(self.doc_ids[j], float(self.item_sim[i, j]))
                for j in range(len(self.doc_ids)) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def recommend_for_user(self, user_id, top_k=5):
        if user_id not in self.matrix.index:
            return []
        user_ratings = self.matrix.loc[user_id]
        rated = user_ratings[user_ratings > 0]
        scores = pd.Series(0.0, index=self.doc_ids)
        for doc_id, rating in rated.items():
            i = self.index[doc_id]
            sim_row = self.item_sim[i]
            scores += sim_row * rating
        scores = scores.drop(labels=rated.index, errors="ignore")
        scores = scores.sort_values(ascending=False)
        return list(scores.head(top_k).items())


class HybridRecommender:
    def __init__(self, content_rec: ContentBasedRecommender, collab_rec: CollaborativeRecommender,
                 alpha=0.5):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.alpha = alpha

    def recommend(self, doc_id, top_k=5):
        content_scores = dict(self.content_rec.recommend(doc_id, top_k=len(self.content_rec.doc_ids)))
        collab_scores = dict(self.collab_rec.recommend_for_item(doc_id, top_k=len(self.collab_rec.doc_ids))) \
            if doc_id in self.collab_rec.index else {}

        all_docs = set(content_scores) | set(collab_scores)
        max_c = max(content_scores.values()) if content_scores else 1.0
        max_v = max(collab_scores.values()) if collab_scores else 1.0
        blended = []
        for d in all_docs:
            c = content_scores.get(d, 0) / (max_c or 1.0)
            v = collab_scores.get(d, 0) / (max_v or 1.0)
            blended.append((d, self.alpha * c + (1 - self.alpha) * v))
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:top_k]
