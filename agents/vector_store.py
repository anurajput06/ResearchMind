"""
In-memory vector store using pure numpy.
No FAISS needed — dot product search on TF-IDF vectors is fast enough
for document sizes we handle (<1000 chunks).
Zero extra RAM overhead vs FAISS (~50MB saved).
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np
from agents.chunking import DocumentChunk


@dataclass
class VectorStore:
    embeddings: np.ndarray        # shape (n_chunks, vocab_dim)
    chunks: List[DocumentChunk]

    @property
    def ntotal(self) -> int:
        return len(self.chunks)


# Thin wrapper so Architecture tab can call .index.ntotal
class _FakeIndex:
    def __init__(self, n): self.ntotal = n

class VectorStoreWithIndex(VectorStore):
    @property
    def index(self): return _FakeIndex(self.ntotal)


class VectorStoreAgent:
    def build_or_load(
        self,
        document_key: str,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> VectorStoreWithIndex:
        if len(chunks) == 0 or embeddings.size == 0:
            raise ValueError("No document text was extracted.")
        return VectorStoreWithIndex(embeddings=embeddings, chunks=chunks)
