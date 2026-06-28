"""
Lightweight embedding agent using fastembed (ONNX runtime).
Uses ~50MB RAM vs ~450MB for sentence-transformers+PyTorch.
Perfect for Render free tier (512MB limit).
"""
from typing import Iterable, List

import numpy as np
import streamlit as st


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    from fastembed import TextEmbedding
    # BAAI/bge-small-en-v1.5 — 384-dim, ~40MB, fast ONNX
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


class EmbeddingAgent:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_embedding_model()
        return self._model

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.empty((0, 384), dtype="float32")
        embeddings = list(self.model.embed(items))
        result = np.array(embeddings, dtype="float32")
        # Normalize for cosine similarity
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (result / norms).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        embeddings = list(self.model.embed([text]))
        result = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (result / norms).astype("float32")
