"""
Text preprocessing and mining framework (Section C).

Provides:
- Multiple preprocessing variants (raw / stopword-removed / stemmed / lemmatized)
  for comparative analysis of strategies.
- Keyword extraction (TF-IDF based).
- Document profiling (length, vocabulary richness, top keywords).
- Simple Naive-Bayes document classification by category.
- Corpus-level statistics used to drive visualizations.
"""

import math
import re
from collections import Counter

from nltk.stem import PorterStemmer

# A bundled, self-contained English stopword list is used instead of NLTK's
# downloadable `stopwords` corpus so the app works out of the box even when
# the run environment (Streamlit / BITS virtual lab / grader's machine) has
# no internet access to fetch NLTK data packages.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "also", "one", "two", "may", "many", "much", "used", "using", "however",
}

STEMMER = PorterStemmer()  # rule-based, no downloaded data required

# Minimal irregular-plural / inflection map so lemmatization still does
# something meaningful without requiring the (network-downloaded) WordNet
# corpus. Falls back to a light suffix-stripping heuristic otherwise.
_IRREGULAR_LEMMAS = {
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "has": "have", "had": "have",
}


def _lemmatize(token):
    if token in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        return token[:-3] + "f"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


TOKEN_RE = re.compile(r"[a-zA-Z]+")


def clean_tokenize(text):
    """Lowercase + alphabetic-only tokenization (simple, dependency-free regex)."""
    tokens = TOKEN_RE.findall(text)
    return [t.lower() for t in tokens]


def preprocessing_variants(text):
    """Return dict of {variant_name: token_list} for comparative analysis."""
    raw_tokens = clean_tokenize(text)
    no_stop = [t for t in raw_tokens if t not in STOPWORDS]
    stemmed = [STEMMER.stem(t) for t in no_stop]
    lemmatized = [_lemmatize(t) for t in no_stop]
    return {
        "raw": raw_tokens,
        "stopword_removed": no_stop,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
    }


def variant_vocab_stats(corpus_texts):
    """Compare vocabulary size / total tokens across preprocessing strategies.

    Returns a list of dict rows suitable for a pandas DataFrame / bar chart.
    """
    rows = []
    for variant in ["raw", "stopword_removed", "stemmed", "lemmatized"]:
        total_tokens = 0
        vocab = set()
        for text in corpus_texts:
            toks = preprocessing_variants(text)[variant]
            total_tokens += len(toks)
            vocab.update(toks)
        rows.append({
            "strategy": variant,
            "total_tokens": total_tokens,
            "vocabulary_size": len(vocab),
            "type_token_ratio": round(len(vocab) / total_tokens, 4) if total_tokens else 0,
        })
    return rows


def extract_keywords_tfidf(doc_tokens, corpus_doc_freq, num_docs, top_n=10):
    """TF-IDF keyword extraction for a single document given corpus document frequencies."""
    tf = Counter(doc_tokens)
    scores = {}
    for term, freq in tf.items():
        df = corpus_doc_freq.get(term, 1)
        idf = math.log((num_docs + 1) / (df + 1)) + 1
        scores[term] = (freq / max(len(doc_tokens), 1)) * idf
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def build_corpus_doc_freq(tokenized_docs):
    """document frequency of each term across the corpus (for IDF/keyword extraction)."""
    df = Counter()
    for tokens in tokenized_docs:
        df.update(set(tokens))
    return df


def document_profile(doc_id, title, tokens, corpus_doc_freq, num_docs):
    keywords = extract_keywords_tfidf(tokens, corpus_doc_freq, num_docs, top_n=8)
    unique_terms = set(tokens)
    return {
        "doc_id": doc_id,
        "title": title,
        "word_count": len(tokens),
        "unique_terms": len(unique_terms),
        "lexical_diversity": round(len(unique_terms) / len(tokens), 4) if tokens else 0,
        "top_keywords": ", ".join(f"{w} ({s:.3f})" for w, s in keywords),
    }


class NaiveBayesClassifier:
    """Minimal multinomial Naive Bayes for document classification by category."""

    def __init__(self):
        self.class_priors = {}
        self.word_likelihoods = {}
        self.vocab = set()
        self.classes = []

    def fit(self, tokenized_docs, labels):
        self.classes = sorted(set(labels))
        class_docs = {c: [] for c in self.classes}
        for tokens, label in zip(tokenized_docs, labels):
            class_docs[label].append(tokens)
            self.vocab.update(tokens)

        total_docs = len(labels)
        for c in self.classes:
            self.class_priors[c] = len(class_docs[c]) / total_docs
            word_counts = Counter()
            for tokens in class_docs[c]:
                word_counts.update(tokens)
            total_words = sum(word_counts.values())
            vocab_size = len(self.vocab)
            self.word_likelihoods[c] = {
                w: (word_counts.get(w, 0) + 1) / (total_words + vocab_size)
                for w in self.vocab
            }
            self.word_likelihoods[c]["__default__"] = 1 / (total_words + vocab_size)

    def predict(self, tokens):
        best_class, best_score = None, -math.inf
        scores = {}
        for c in self.classes:
            score = math.log(self.class_priors[c])
            likelihoods = self.word_likelihoods[c]
            default = likelihoods["__default__"]
            for t in tokens:
                score += math.log(likelihoods.get(t, default))
            scores[c] = score
            if score > best_score:
                best_class, best_score = c, score
        return best_class, scores

    def evaluate(self, tokenized_docs, labels):
        correct = 0
        predictions = []
        for tokens, true_label in zip(tokenized_docs, labels):
            pred, _ = self.predict(tokens)
            predictions.append(pred)
            correct += int(pred == true_label)
        accuracy = correct / len(labels) if labels else 0
        return accuracy, predictions
