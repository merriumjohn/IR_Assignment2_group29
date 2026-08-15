"""
Evaluation metrics (Section F).

precision, recall, f1, precision@k, recall@k, average precision (for MAP),
reciprocal rank (for MRR) and NDCG@k.
"""

import math


def precision_recall_f1(retrieved, relevant):
    retrieved_set, relevant_set = set(retrieved), set(relevant)
    tp = len(retrieved_set & relevant_set)
    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def precision_at_k(ranked_list, relevant, k):
    top_k = ranked_list[:k]
    relevant_set = set(relevant)
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / k if k else 0.0


def recall_at_k(ranked_list, relevant, k):
    top_k = ranked_list[:k]
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / len(relevant_set)


def average_precision(ranked_list, relevant):
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits, sum_precisions = 0, 0.0
    for i, doc in enumerate(ranked_list, start=1):
        if doc in relevant_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant_set) if hits else 0.0


def reciprocal_rank(ranked_list, relevant):
    relevant_set = set(relevant)
    for i, doc in enumerate(ranked_list, start=1):
        if doc in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_list, relevant, k):
    relevant_set = set(relevant)
    dcg = 0.0
    for i, doc in enumerate(ranked_list[:k], start=1):
        rel = 1 if doc in relevant_set else 0
        dcg += rel / math.log2(i + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_query(ranked_list, relevant, k=5):
    precision, recall, f1 = precision_recall_f1(ranked_list, relevant)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        f"precision@{k}": round(precision_at_k(ranked_list, relevant, k), 4),
        f"recall@{k}": round(recall_at_k(ranked_list, relevant, k), 4),
        "average_precision": round(average_precision(ranked_list, relevant), 4),
        "reciprocal_rank": round(reciprocal_rank(ranked_list, relevant), 4),
        f"ndcg@{k}": round(ndcg_at_k(ranked_list, relevant, k), 4),
    }


def evaluate_all(query_results, k=5):
    """query_results: list of (query, ranked_doc_ids, relevant_doc_ids).
    Returns (per_query_rows, aggregate_dict) with MAP and MRR across queries.
    """
    rows = []
    for query, ranked, relevant in query_results:
        metrics = evaluate_query(ranked, relevant, k=k)
        metrics["query"] = query
        rows.append(metrics)

    if not rows:
        return [], {}

    aggregate = {
        "MAP": round(sum(r["average_precision"] for r in rows) / len(rows), 4),
        "MRR": round(sum(r["reciprocal_rank"] for r in rows) / len(rows), 4),
        "mean_precision": round(sum(r["precision"] for r in rows) / len(rows), 4),
        "mean_recall": round(sum(r["recall"] for r in rows) / len(rows), 4),
        "mean_f1": round(sum(r["f1"] for r in rows) / len(rows), 4),
        f"mean_ndcg@{k}": round(sum(r[f"ndcg@{k}"] for r in rows) / len(rows), 4),
    }
    return rows, aggregate
