import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np

from agents.chunking import DocumentChunk
from config import CACHE_DIR


@dataclass
class VectorStore:
    index: faiss.Index
    chunks: List[DocumentChunk]


class VectorStoreAgent:
    def build_or_load(self, document_key: str, chunks: List[DocumentChunk], embeddings: np.ndarray) -> VectorStore:
        CACHE_DIR.mkdir(exist_ok=True)
        index_path = CACHE_DIR / f"{document_key}.index"
        chunks_path = CACHE_DIR / f"{document_key}.chunks.pkl"

        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with chunks_path.open("rb") as handle:
                cached_chunks = pickle.load(handle)
            return VectorStore(index=index, chunks=cached_chunks)

        if len(chunks) == 0 or embeddings.size == 0:
            raise ValueError("No document text was extracted from this PDF.")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, str(index_path))
        with chunks_path.open("wb") as handle:
            pickle.dump(chunks, handle)

        return VectorStore(index=index, chunks=chunks)
