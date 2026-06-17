"""Retrieve relevant scientific-paper chunks from a FAISS index."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_PATH = Path("data/faiss.index")
DEFAULT_METADATA_PATH = Path("data/chunk_metadata.json")
DEFAULT_MODEL_NAME = "BAAI/bge-m3"


def choose_device() -> str:
    """Choose CUDA for query embedding when it is available."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class Retriever:
    """Load retrieval artifacts once and answer multiple retrieval queries."""

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Load the FAISS index, ordered metadata, and embedding model."""
        self.index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, list) or not all(
            isinstance(chunk, Mapping) for chunk in metadata
        ):
            raise ValueError(f"Expected a JSON list of chunk objects in {metadata_path}")
        if self.index.ntotal != len(metadata):
            raise ValueError(
                "FAISS vector count does not match chunk metadata count: "
                f"{self.index.ntotal} != {len(metadata)}"
            )

        self.metadata = [dict(chunk) for chunk in metadata]
        self.model = SentenceTransformer(model_name, device=choose_device())

    def _encode_question(self, question: str) -> np.ndarray:
        """Encode one normalized query vector."""
        if not question.strip():
            raise ValueError("question must not be empty")
        embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(embedding, dtype=np.float32)

    def _search_ids(
        self, query_embedding: np.ndarray, candidate_ids: list[int], top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search globally or over an exact subset of vector IDs."""
        if len(candidate_ids) == self.index.ntotal:
            scores, ids = self.index.search(query_embedding, top_k)
            return scores[0], ids[0]

        candidate_vectors = np.vstack(
            [self.index.reconstruct(vector_id) for vector_id in candidate_ids]
        ).astype(np.float32)
        subset_index = faiss.IndexFlatIP(self.index.d)
        subset_index.add(candidate_vectors)
        scores, subset_ids = subset_index.search(query_embedding, top_k)
        original_ids = np.array(
            [candidate_ids[subset_id] if subset_id >= 0 else -1 for subset_id in subset_ids[0]],
            dtype=np.int64,
        )
        return scores[0], original_ids

    def retrieve(
        self, question: str, paper_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the most similar chunks, optionally limited to one paper."""
        if top_k < 1:
            raise ValueError("top_k must be greater than 0")

        candidate_ids = [
            index
            for index, chunk in enumerate(self.metadata)
            if paper_id is None or str(chunk.get("paper_id")) == str(paper_id)
        ]
        if not candidate_ids:
            return []

        result_count = min(top_k, len(candidate_ids))
        query_embedding = self._encode_question(question)
        scores, ids = self._search_ids(query_embedding, candidate_ids, result_count)

        results: list[dict[str, Any]] = []
        for score, vector_id in zip(scores, ids):
            if vector_id < 0:
                continue
            chunk = self.metadata[int(vector_id)]
            results.append(
                {
                    "paper_id": chunk.get("paper_id", ""),
                    "title": chunk.get("title", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": float(score),
                    "text": chunk.get("text", ""),
                }
            )
        return results


_DEFAULT_RETRIEVER: Retriever | None = None


def retrieve(
    question: str, paper_id: str | None = None, top_k: int = 5
) -> list[dict[str, Any]]:
    """Retrieve chunks using the default model and artifacts.

    The default retriever is loaded lazily and cached for subsequent calls.
    Use ``Retriever`` directly when custom model or artifact paths are needed.
    """
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = Retriever()
    return _DEFAULT_RETRIEVER.retrieve(question, paper_id=paper_id, top_k=top_k)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a retrieval test query."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--paper_id")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--index_path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata_path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    """Run a retrieval query and print JSON results."""
    args = parse_args()
    retriever = Retriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        model_name=args.model_name,
    )
    results = retriever.retrieve(args.question, paper_id=args.paper_id, top_k=args.top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
