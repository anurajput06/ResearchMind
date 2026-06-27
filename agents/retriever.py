"""
Retriever using dot product similarity on numpy arrays.
Works with the in-memory VectorStore (no FAISS required).
"""
from dataclasses import dataclass
from typing import List
import numpy as np
from agents.vector_store import VectorStoreWithIndex


@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_id: int


class RetrieverAgent:
    def retrieve(
        self,
        vector_store: VectorStoreWithIndex,
        query_embedding: np.ndarray,
        top_k: int = 4,
    ) -> List[RetrievedChunk]:
        n = vector_store.ntotal
        if n == 0:
            return []
        k = min(top_k, n)

        # Dot product = cosine similarity (both embeddings are L2-normalised)
        scores = (vector_store.embeddings @ query_embedding.T).squeeze()

        if scores.ndim == 0:
            scores = np.array([float(scores)])

        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            chunk = vector_store.chunks[int(idx)]
            results.append(RetrievedChunk(
                text=chunk.text,
                score=float(scores[idx]),
                chunk_id=chunk.id,
            ))
        return results
