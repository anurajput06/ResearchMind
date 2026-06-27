from typing import Iterable, List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@st.cache_resource(show_spinner="Loading embedding model")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


class EmbeddingAgent:
    def __init__(self):
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = load_embedding_model()
        return self._model

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.empty((0, 384), dtype="float32")
        embeddings = self.model.encode(
            items,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype("float32")
