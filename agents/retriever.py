from dataclasses import dataclass
from typing import List

import numpy as np

from agents.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_id: int


class RetrieverAgent:
    def retrieve(self, vector_store: VectorStore, query_embedding: np.ndarray, top_k: int = 4) -> List[RetrievedChunk]:
        k = min(top_k, vector_store.index.ntotal)
        if k <= 0:
            return []

        scores, indexes = vector_store.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indexes[0]):
            if idx == -1:
                continue
            chunk = vector_store.chunks[int(idx)]
            results.append(RetrievedChunk(text=chunk.text, score=float(score), chunk_id=chunk.id))
        return results
