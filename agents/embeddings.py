"""
Memory-safe embedding agent using TF-IDF (scikit-learn).
RAM usage: ~5MB vs 450MB for sentence-transformers+torch.
No PyTorch, no ONNX, no GPU libraries — runs fine on Render free tier.
"""
from typing import Iterable, List
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache_resource(show_spinner="Initialising search index…")
def load_vectorizer():
    return TfidfVectorizer(
        max_features=8192,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )


class EmbeddingAgent:
    """
    Stateful TF-IDF embedder.
    Must call embed_documents before embed_query so the vocab is fitted.
    """
    def __init__(self):
        self._vectorizer = None
        self._fitted = False

    def _get_vectorizer(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                max_features=8192,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                strip_accents="unicode",
            )
        return self._vectorizer

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.empty((0, 1), dtype="float32")
        vec = self._get_vectorizer()
        matrix = vec.fit_transform(items).toarray().astype("float32")
        self._fitted = True
        # L2 normalise for cosine similarity via dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return matrix / norms

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_documents before embed_query.")
        vec = self._get_vectorizer()
        q = vec.transform([text]).toarray().astype("float32")
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        return q
