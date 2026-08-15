"""
Information Retrieval - Assignment 2
Merged Course: AIMLCZG537 / DSECLZG537 (S2-25)
Group 29

End-to-end Streamlit application covering: crawling, text mining, indexing,
web search & ranking (TF-IDF + PageRank/HITS), recommendation (content-based,
collaborative, hybrid) and IR evaluation metrics - all driven from the UI,
no separate notebooks or backend scripts required at run time.
"""

import json
import os
import textwrap
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from ir_engine.crawler import WebCrawler, load_local_dataset
from ir_engine.preprocessing import (
    NaiveBayesClassifier,
    build_corpus_doc_freq,
    clean_tokenize,
    document_profile,
    preprocessing_variants,
    variant_vocab_stats,
)
from ir_engine.indexing import InvertedIndex
from ir_engine.ranking import VectorSpaceSearch, combined_ranking, hits, page_rank
from ir_engine.recommender import CollaborativeRecommender, ContentBasedRecommender, HybridRecommender
from ir_engine.evaluation import evaluate_all

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ------------------------------------------------------------------
# Session-state bootstrap
# ------------------------------------------------------------------

def log_perf(stage, start, **extra):
    elapsed = time.perf_counter() - start
    entry = {"stage": stage, "seconds": round(elapsed, 4),
             "timestamp": time.strftime("%H:%M:%S"), **extra}
    st.session_state.perf_log.append(entry)
    return elapsed


def init_state():
    if st.session_state.get("initialized"):
        return

    start = time.perf_counter()
    documents, metadata = load_local_dataset(os.path.join(DATA_DIR, "articles.json"))

    edges = []
    edges_path = os.path.join(DATA_DIR, "link_graph.json")
    if os.path.exists(edges_path):
        with open(edges_path, "r", encoding="utf-8") as f:
            edges = [tuple(e) for e in json.load(f)]

    interactions_df = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))

    judgments = {}
    judgments_path = os.path.join(DATA_DIR, "relevance_judgments.json")
    if os.path.exists(judgments_path):
        with open(judgments_path, "r", encoding="utf-8") as f:
            judgments = json.load(f)

    st.session_state.documents = documents
    st.session_state.metadata = metadata
    st.session_state.link_edges = edges
    st.session_state.interactions_df = interactions_df
    st.session_state.relevance_judgments = judgments
    st.session_state.perf_log = []
    st.session_state.index = None
    st.session_state.vss = None
    st.session_state.pagerank_scores = {}
    st.session_state.hits_scores = ({}, {})
    st.session_state.crawl_history = []
    st.session_state.initialized = True

    log_perf("bootstrap_load_dataset", start, num_docs=len(documents))
    build_index_and_search()


def build_index_and_search():
    start = time.perf_counter()
    docs = st.session_state.documents
    tokenized = {doc_id: preprocessing_variants(d["content"])["stopword_removed"]
                 for doc_id, d in docs.items()}
    content_hashes = {doc_id: st.session_state.metadata[doc_id].get("content_hash")
                      for doc_id in docs}

    index = InvertedIndex()
    index.build(tokenized, content_hashes)
    st.session_state.index = index

    doc_ids = list(docs.keys())
    texts = [docs[d]["content"] for d in doc_ids]
    st.session_state.vss = VectorSpaceSearch(doc_ids, texts)
    st.session_state.doc_ids_ordered = doc_ids

    log_perf("build_index_and_vectorizer", start, num_docs=len(docs))


def corpus_dataframe():
    docs, meta = st.session_state.documents, st.session_state.metadata
    rows = []
    for doc_id, d in docs.items():
        m = meta.get(doc_id, {})
        rows.append({
            "doc_id": doc_id,
            "title": d["title"],
            "category": m.get("category", "Uncategorized"),
            "url": m.get("url", ""),
            "source": m.get("source", ""),
            "word_count": len(d["content"].split()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

def page_dashboard():
    st.header("Dashboard")
    df = corpus_dataframe()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(df))
    c2.metric("Categories", df["category"].nunique())
    c3.metric("Avg. words/doc", int(df["word_count"].mean()) if len(df) else 0)
    idx_terms = len(st.session_state.index.postings) if st.session_state.index else 0
    c4.metric("Index terms", idx_terms)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average word count per category")
        avg_words = df.groupby("category")["word_count"].mean().reset_index()
        avg_words = avg_words.sort_values("word_count", ascending=True)
        fig = px.bar(avg_words, x="word_count", y="category", orientation="h",
                     color="word_count", color_continuous_scale="Blues",
                     text="word_count", title="Average word count per category")
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Word count distribution by category")
        fig2 = px.box(df, x="category", y="word_count", color="category",
                      points="all", title="Word count distribution by category")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Source mix")
    st.dataframe(df["source"].value_counts().rename_axis("source").reset_index(name="count"),
                 use_container_width=True)

    st.subheader("Corpus preview")
    st.dataframe(df, use_container_width=True, height=300)


def page_crawling():
    st.header("Crawling Interface")
    st.caption("Configurable seed sources and crawl depth. Duplicate URLs and "
               "duplicate documents are detected and skipped. Metadata is "
               "stored separately from document content.")

    default_seeds = "https://en.wikipedia.org/wiki/Natural_language_processing\nhttps://en.wikipedia.org/wiki/Information_retrieval"
    seeds_text = st.text_area("Seed URLs (one per line)", value=default_seeds, height=100)
    col1, col2, col3 = st.columns(3)
    max_depth = col1.slider("Max crawl depth", 0, 3, 1)
    max_pages = col2.slider("Max pages", 1, 60, 15)
    link_limit = col3.slider("Outlinks followed per page", 1, 10, 4)
    category_label = st.text_input("Category label to assign to newly crawled docs", value="Uncategorized")

    if st.button("Start Crawl", type="primary"):
        seeds = [s.strip() for s in seeds_text.splitlines() if s.strip()]
        if not seeds:
            st.warning("Provide at least one seed URL.")
            return
        start = time.perf_counter()
        with st.spinner(f"Crawling {len(seeds)} seed source(s) at depth {max_depth}..."):
            crawler = WebCrawler(max_depth=max_depth, max_pages=max_pages,
                                  per_page_link_limit=link_limit)
            try:
                docs, metadata, edges = crawler.crawl(seeds)
            except Exception as e:
                st.error(f"Crawl failed: {e}")
                return
        elapsed = log_perf("crawl", start, pages_fetched=len(docs))

        if not docs:
            st.warning("No pages were fetched. Check network access / seed URLs "
                       "(live crawling requires internet access from the run environment).")
            return

        existing = len(st.session_state.documents)
        remapped_docs, remapped_meta = {}, {}
        for i, (doc_id, doc) in enumerate(docs.items()):
            new_id = f"crawl_{existing + i:04d}"
            remapped_docs[new_id] = doc
            m = metadata[doc_id]
            m["category"] = category_label
            m["source"] = "live_crawl"
            remapped_meta[new_id] = m

        st.session_state.documents.update(remapped_docs)
        st.session_state.metadata.update(remapped_meta)
        st.session_state.crawl_history.append({
            "seeds": seeds, "depth": max_depth, "pages": len(docs),
            "seconds": round(elapsed, 2), "stats": crawler.stats,
        })

        st.success(f"Fetched {len(docs)} page(s) in {elapsed:.2f}s. "
                   f"Duplicate URLs skipped: {crawler.stats['duplicate_urls']}, "
                   f"duplicate content skipped: {crawler.stats['duplicate_content']}, "
                   f"errors: {crawler.stats['errors']}.")
        st.dataframe(pd.DataFrame([
            {"title": d["title"], "words": len(d["content"].split()),
             "depth": metadata[k]["crawl_depth"], "url": metadata[k]["url"]}
            for k, d in docs.items()
        ]), use_container_width=True)
        st.info("Rebuild the index on the **Index Management** page to include these new documents in search.")

    if st.session_state.crawl_history:
        st.subheader("Crawl history (this session)")
        st.dataframe(pd.DataFrame(st.session_state.crawl_history), use_container_width=True)


def page_text_mining():
    st.header("Text Preprocessing & Mining")
    docs = st.session_state.documents

    st.subheader("Comparative preprocessing strategy analysis")
    if st.button("Compute vocabulary comparison across strategies"):
        start = time.perf_counter()
        texts = [d["content"] for d in docs.values()]
        rows = variant_vocab_stats(texts)
        log_perf("preprocessing_comparison", start, num_docs=len(texts))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        fig = px.bar(df, x="strategy", y=["total_tokens", "vocabulary_size"],
                     barmode="group", title="Tokens & vocabulary size by preprocessing strategy")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Document profiling & keyword extraction")
    tokenized_docs = {doc_id: preprocessing_variants(d["content"])["stopword_removed"]
                       for doc_id, d in docs.items()}
    doc_freq = build_corpus_doc_freq(list(tokenized_docs.values()))
    num_docs = len(docs)

    if st.button("Build document profiles (TF-IDF keywords)"):
        start = time.perf_counter()
        rows = [document_profile(doc_id, docs[doc_id]["title"], tokenized_docs[doc_id], doc_freq, num_docs)
                for doc_id in docs]
        log_perf("document_profiling", start, num_docs=num_docs)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

    st.subheader("Single document inspection")
    doc_id = st.selectbox("Choose a document", list(docs.keys()),
                          format_func=lambda d: docs[d]["title"])
    if doc_id:
        variants = preprocessing_variants(docs[doc_id]["content"])
        cols = st.columns(4)
        for col, (name, tokens) in zip(cols, variants.items()):
            col.metric(name, len(tokens))
        with st.expander("Show raw content"):
            st.write(docs[doc_id]["content"][:2000])

    st.subheader("Document classification (Naive Bayes by category)")
    if st.button("Train / evaluate category classifier"):
        start = time.perf_counter()
        meta = st.session_state.metadata
        doc_ids = list(docs.keys())

        # Stratified 70/30 split per category so every class is represented
        # in both train and test (a plain sequential split would starve
        # categories that happen to sit at the tail of the corpus).
        by_category = {}
        for d in doc_ids:
            by_category.setdefault(meta[d].get("category", "Uncategorized"), []).append(d)

        train_ids, test_ids = [], []
        for cat, ids in by_category.items():
            split = max(1, int(0.7 * len(ids)))
            train_ids.extend(ids[:split])
            test_ids.extend(ids[split:] or ids[-1:])

        if not test_ids:
            st.warning("Not enough documents for a train/test split.")
        else:
            clf = NaiveBayesClassifier()
            clf.fit([tokenized_docs[d] for d in train_ids], [meta[d]["category"] for d in train_ids])
            accuracy, preds = clf.evaluate([tokenized_docs[d] for d in test_ids],
                                            [meta[d]["category"] for d in test_ids])
            log_perf("classification", start, train=len(train_ids), test=len(test_ids))
            st.metric("Test accuracy", f"{accuracy:.2%}")
            result_df = pd.DataFrame({
                "title": [docs[d]["title"] for d in test_ids],
                "true_category": [meta[d]["category"] for d in test_ids],
                "predicted_category": preds,
            })
            st.dataframe(result_df, use_container_width=True)


def page_index_management():
    st.header("Index Management")
    st.caption("Duplicate documents (by content hash) are detected and excluded before indexing.")

    variant = st.selectbox("Preprocessing variant used for indexing",
                            ["stopword_removed", "raw", "stemmed", "lemmatized"], index=0)

    if st.button("(Re)build inverted index", type="primary"):
        start = time.perf_counter()
        docs = st.session_state.documents
        tokenized = {doc_id: preprocessing_variants(d["content"])[variant] for doc_id, d in docs.items()}
        content_hashes = {doc_id: st.session_state.metadata[doc_id].get("content_hash") for doc_id in docs}
        index = InvertedIndex()
        index.build(tokenized, content_hashes)
        st.session_state.index = index
        st.session_state.vss = VectorSpaceSearch(list(docs.keys()), [docs[d]["content"] for d in docs])
        st.session_state.doc_ids_ordered = list(docs.keys())
        log_perf("manual_index_rebuild", start, num_docs=len(docs), variant=variant)
        st.success("Index rebuilt.")

    index = st.session_state.index
    if index is not None:
        stats = index.stats()
        cols = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, v)

        st.subheader("Postings lookup")
        term = st.text_input("Enter a term to inspect its postings list")
        if term:
            postings = index.postings.get(term.lower().strip(), {})
            if postings:
                docs = st.session_state.documents
                rows = [{"doc_id": d, "title": docs[d]["title"], "term_frequency": tf}
                        for d, tf in sorted(postings.items(), key=lambda x: -x[1])]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Term not found in the index.")

        st.download_button("Download index as JSON", index.to_json(),
                            file_name="inverted_index.json", mime="application/json")
    else:
        st.info("Build the index to see statistics.")


def page_search():
    st.header("Web Search")
    if st.session_state.vss is None:
        st.warning("Build the index first (Index Management page).")
        return

    query = st.text_input("Search query")
    col1, col2, col3 = st.columns(3)
    top_k = col1.slider("Top-K results", 1, 20, 10)
    use_index_optimization = col2.checkbox("Use inverted index for query optimization", value=True)
    ranking_mode = col3.selectbox("Ranking signal", ["TF-IDF only", "TF-IDF + PageRank", "TF-IDF + HITS authority"])

    alpha = 0.7
    if ranking_mode != "TF-IDF only":
        alpha = st.slider("Weight on TF-IDF similarity (alpha)", 0.0, 1.0, 0.7)

    if query and st.button("Search", type="primary"):
        start = time.perf_counter()
        vss = st.session_state.vss
        candidates = None
        if use_index_optimization and st.session_state.index is not None:
            candidates = vss.candidate_docs_from_index(query, st.session_state.index)

        tfidf_results = vss.search(query, top_k=max(top_k, 50), candidate_doc_ids=candidates)

        if ranking_mode == "TF-IDF only":
            results = [(d, s, s, 0.0) for d, s in tfidf_results][:top_k]
        else:
            if ranking_mode == "TF-IDF + PageRank":
                if not st.session_state.pagerank_scores:
                    st.session_state.pagerank_scores = page_rank(st.session_state.doc_ids_ordered,
                                                                   st.session_state.link_edges)
                link_scores = st.session_state.pagerank_scores
            else:
                if not any(st.session_state.hits_scores):
                    st.session_state.hits_scores = hits(st.session_state.doc_ids_ordered,
                                                          st.session_state.link_edges)
                link_scores = st.session_state.hits_scores[1]
            results = combined_ranking(tfidf_results, link_scores, alpha=alpha)[:top_k]

        elapsed = log_perf("search", start, query=query, mode=ranking_mode, results=len(results))

        docs, meta = st.session_state.documents, st.session_state.metadata
        rows = []
        for rank, (doc_id, score, sim, link_score) in enumerate(results, start=1):
            content = docs[doc_id]["content"]
            rows.append({
                "rank": rank,
                "title": docs[doc_id]["title"],
                "category": meta[doc_id].get("category", "Uncategorized"),
                "combined_score": round(score, 4),
                "tfidf_similarity": round(sim, 4),
                "link_score": round(link_score, 4),
                "snippet": content[:180] + "...",
                "url": meta[doc_id].get("url", ""),
            })
        st.caption(f"{len(results)} result(s) in {elapsed:.3f}s")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)


def page_ranking_visualization():
    st.header("Ranking Visualization")
    st.caption("PageRank / HITS computed over the real hyperlink graph discovered during crawling.")

    if st.button("Compute PageRank & HITS", type="primary"):
        start = time.perf_counter()
        doc_ids = st.session_state.doc_ids_ordered
        edges = st.session_state.link_edges
        st.session_state.pagerank_scores = page_rank(doc_ids, edges)
        st.session_state.hits_scores = hits(doc_ids, edges)
        log_perf("pagerank_hits_compute", start, num_docs=len(doc_ids), num_edges=len(edges))
        st.success(f"Computed over {len(doc_ids)} documents and {len(edges)} link edges.")

    docs = st.session_state.documents
    pr = st.session_state.pagerank_scores
    hub, auth = st.session_state.hits_scores

    if pr or auth:
        col1, col2 = st.columns(2)
        with col1:
            if pr:
                top_pr = sorted(pr.items(), key=lambda x: -x[1])[:15]
                df_pr = pd.DataFrame([{"title": docs[d]["title"], "pagerank": s} for d, s in top_pr])
                st.plotly_chart(px.bar(df_pr, x="pagerank", y="title", orientation="h",
                                        title="Top-15 PageRank"), use_container_width=True)
            else:
                st.info("Click 'Compute PageRank & HITS' above to see PageRank scores.")
        with col2:
            if auth:
                top_auth = sorted(auth.items(), key=lambda x: -x[1])[:15]
                df_auth = pd.DataFrame([{"title": docs[d]["title"], "authority": s} for d, s in top_auth])
                st.plotly_chart(px.bar(df_auth, x="authority", y="title", orientation="h",
                                        title="Top-15 HITS authority"), use_container_width=True)
            else:
                st.info("Click 'Compute PageRank & HITS' above to see HITS authority scores.")

    st.subheader("How link-based ranking changes result order")
    query = st.text_input("Query to compare rankings")
    alpha = st.slider("Weight on TF-IDF (alpha) for the blended ranking", 0.0, 1.0, 0.6)
    if query and pr and st.session_state.vss is not None:
        tfidf_results = st.session_state.vss.search(query, top_k=15)
        blended = combined_ranking(tfidf_results, pr, alpha=alpha)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pure TF-IDF ranking**")
            st.dataframe(pd.DataFrame([
                {"rank": i + 1, "title": docs[d]["title"], "score": round(s, 4)}
                for i, (d, s) in enumerate(tfidf_results)
            ]), use_container_width=True)
        with c2:
            st.markdown("**Blended TF-IDF + PageRank ranking**")
            st.dataframe(pd.DataFrame([
                {"rank": i + 1, "title": docs[d]["title"], "score": round(s, 4)}
                for i, (d, s, _, _) in enumerate(blended)
            ]), use_container_width=True)
    elif query and not pr:
        st.info("Compute PageRank & HITS above first.")


def page_recommendations():
    st.header("Recommendation Panel")
    docs = st.session_state.documents
    doc_ids = list(docs.keys())

    top_k = st.slider("Top-K recommendations", 1, 15, 5)
    selected_doc = st.selectbox("Base article", doc_ids, format_func=lambda d: docs[d]["title"])

    tab1, tab2, tab3 = st.tabs(["Content-based", "Collaborative", "Hybrid"])

    with tab1:
        start = time.perf_counter()
        content_rec = ContentBasedRecommender(st.session_state.doc_ids_ordered,
                                               st.session_state.vss.doc_matrix)
        recs = content_rec.recommend(selected_doc, top_k=top_k)
        log_perf("content_based_recommend", start, base_doc=selected_doc)
        st.dataframe(pd.DataFrame([
            {"title": docs[d]["title"], "similarity_score": round(s, 4)} for d, s in recs
        ]), use_container_width=True)

    with tab2:
        interactions_df = st.session_state.interactions_df
        collab_rec = CollaborativeRecommender(interactions_df)
        start = time.perf_counter()
        recs = collab_rec.recommend_for_item(selected_doc, top_k=top_k)
        log_perf("collaborative_item_recommend", start, base_doc=selected_doc)
        if recs:
            st.dataframe(pd.DataFrame([
                {"title": docs[d]["title"], "similarity_score": round(s, 4)} for d, s in recs
            ]), use_container_width=True)
        else:
            st.info("No interaction data for this document yet.")

        st.markdown("---")
        st.subheader("Recommend for a simulated user")
        user_id = st.selectbox("User", sorted(interactions_df["user_id"].unique()))
        user_recs = collab_rec.recommend_for_user(user_id, top_k=top_k)
        st.dataframe(pd.DataFrame([
            {"title": docs[d]["title"], "predicted_score": round(s, 4)} for d, s in user_recs
        ]), use_container_width=True)

    with tab3:
        alpha = st.slider("Weight: content-based vs collaborative (alpha)", 0.0, 1.0, 0.5)
        content_rec = ContentBasedRecommender(st.session_state.doc_ids_ordered, st.session_state.vss.doc_matrix)
        collab_rec = CollaborativeRecommender(st.session_state.interactions_df)
        hybrid = HybridRecommender(content_rec, collab_rec, alpha=alpha)
        start = time.perf_counter()
        recs = hybrid.recommend(selected_doc, top_k=top_k)
        log_perf("hybrid_recommend", start, base_doc=selected_doc)
        st.dataframe(pd.DataFrame([
            {"title": docs[d]["title"], "hybrid_score": round(s, 4)} for d, s in recs
        ]), use_container_width=True)


def page_evaluation():
    st.header("Evaluation Dashboard")
    judgments = st.session_state.relevance_judgments
    if not judgments:
        st.warning("No relevance judgments available.")
        return

    k = st.slider("k (for Precision@k / Recall@k / NDCG@k)", 1, 20, 10)
    ranking_mode = st.selectbox("Ranking configuration to evaluate",
                                 ["TF-IDF only", "TF-IDF + PageRank"])

    if st.button("Run evaluation", type="primary"):
        start = time.perf_counter()
        vss = st.session_state.vss
        pr = st.session_state.pagerank_scores or page_rank(st.session_state.doc_ids_ordered,
                                                            st.session_state.link_edges)
        st.session_state.pagerank_scores = pr

        query_results = []
        for query, relevant in judgments.items():
            tfidf_results = vss.search(query, top_k=len(st.session_state.doc_ids_ordered))
            if ranking_mode == "TF-IDF only":
                ranked = [d for d, _ in tfidf_results]
            else:
                blended = combined_ranking(tfidf_results, pr, alpha=0.7)
                ranked = [d for d, _, _, _ in blended]
            query_results.append((query, ranked, relevant))

        rows, aggregate = evaluate_all(query_results, k=k)
        log_perf("evaluation_run", start, num_queries=len(judgments), mode=ranking_mode)

        st.subheader("Aggregate metrics")
        cols = st.columns(len(aggregate))
        for col, (name, val) in zip(cols, aggregate.items()):
            col.metric(name, val)

        st.subheader("Per-query metrics")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.subheader("Comparative visualization")
        metric_cols = [c for c in df.columns if c != "query"]
        wrapped_queries = ["<br>".join(textwrap.wrap(q, width=18)) for q in df["query"]]
        pivot = df.set_index("query")[metric_cols].T
        pivot.columns = wrapped_queries
        fig = px.imshow(
            pivot,
            color_continuous_scale="Blues",
            aspect="auto",
            title=f"Per-query IR metrics heatmap ({ranking_mode})",
            text_auto=".2f",
        )
        fig.update_xaxes(tickangle=0, tickfont=dict(size=9))
        fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        st.session_state[f"eval_{ranking_mode}"] = (rows, aggregate)

    tfidf_res = st.session_state.get("eval_TF-IDF only")
    combined_res = st.session_state.get("eval_TF-IDF + PageRank")
    if tfidf_res and combined_res:
        st.subheader("Comparative analysis: TF-IDF only vs TF-IDF + PageRank")
        comp_df = pd.DataFrame([
            {"configuration": "TF-IDF only", **tfidf_res[1]},
            {"configuration": "TF-IDF + PageRank", **combined_res[1]},
        ])
        st.dataframe(comp_df, use_container_width=True)
        melted = comp_df.melt(id_vars="configuration", var_name="metric", value_name="value")
        fig = px.bar(melted, x="metric", y="value", color="configuration", barmode="group",
                     title="Aggregate metric comparison")
        st.plotly_chart(fig, use_container_width=True)


def page_performance_analytics():
    st.header("Performance Analytics")
    log = st.session_state.perf_log
    if not log:
        st.info("No operations logged yet - interact with other pages first.")
        return

    df = pd.DataFrame(log)
    st.dataframe(df, use_container_width=True, height=300)

    df["stage_display"] = df["stage"].apply(
        lambda s: "<br>".join(textwrap.wrap(s.replace("_", " "), width=24))
    )
    plot_df = df.groupby("stage", as_index=False).agg({"seconds": "sum", "stage_display": "first"})
    plot_df = plot_df.sort_values("seconds", ascending=False)
    blue_scale = [
        [0.0, "#ffffff"], [0.2, "#d9e9f7"], [0.4, "#9ecae1"],
        [0.6, "#4292c6"], [0.8, "#2171b5"], [1.0, "#08306b"]
    ]
    fig = px.bar(plot_df, x="seconds", y="stage_display", orientation="h",
                 color="seconds", color_continuous_scale=blue_scale,
                 title="Time spent per operation")
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    if "num_docs" in df.columns:
        growth = df.dropna(subset=["num_docs"])[["timestamp", "stage", "stage_display", "num_docs"]]
        if not growth.empty:
            st.subheader("Corpus size over session operations")
            st.plotly_chart(px.line(growth, x="timestamp", y="num_docs", markers=True,
                                     hover_name="stage_display", title="Corpus size (documents) over time"),
                             use_container_width=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    st.set_page_config(page_title="IR System - Assignment 2", page_icon="🔎", layout="wide")
    init_state()

    st.sidebar.title("📌 Navigation")
    st.sidebar.caption("Information Retrieval - Assignment 2 - Group 29")
    page = st.sidebar.radio("Select Section:", [
        "Dashboard",
        "Crawling Interface",
        "Text Preprocessing & Mining",
        "Index Management",
        "Search",
        "Ranking Visualization",
        "Recommendation Panel",
        "Evaluation Dashboard",
        "Performance Analytics",
    ])
    st.sidebar.markdown("---")
    st.sidebar.success(f"{len(st.session_state.documents)} documents in corpus")
    st.sidebar.caption(f"{len(st.session_state.link_edges)} link-graph edges")

    st.title("🔎 InfoSeek: Multi-Source News Intelligence & Retrieval System")

    pages = {
        "Dashboard": page_dashboard,
        "Crawling Interface": page_crawling,
        "Text Preprocessing & Mining": page_text_mining,
        "Index Management": page_index_management,
        "Search": page_search,
        "Ranking Visualization": page_ranking_visualization,
        "Recommendation Panel": page_recommendations,
        "Evaluation Dashboard": page_evaluation,
        "Performance Analytics": page_performance_analytics,
    }
    pages[page]()


if __name__ == "__main__":
    main()
