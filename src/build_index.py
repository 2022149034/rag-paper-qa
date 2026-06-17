"""Build a FAISS cosine-similarity index over scientific-paper chunks."""

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


DEFAULT_CHUNKS_PATH = Path("data/chunks.json")
DEFAULT_INDEX_PATH = Path("data/faiss.index")
DEFAULT_METADATA_PATH = Path("data/chunk_metadata.json")
DEFAULT_MODEL_NAME = "BAAI/bge-m3"


def choose_device() -> str:
    """Choose CUDA for embedding generation when it is available."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load chunks and validate the fields required by retrieval."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    chunks: list[dict[str, Any]] = []
    required_fields = ("paper_id", "title", "chunk_id", "text")
    for index, chunk in enumerate(data):
        if not isinstance(chunk, Mapping):
            raise ValueError(f"Chunk {index} is not a JSON object")
        missing = [field for field in required_fields if field not in chunk]
        if missing:
            raise ValueError(f"Chunk {index} is missing fields: {', '.join(missing)}")
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            continue
        chunks.append(dict(chunk))

    if not chunks:
        raise ValueError(f"No non-empty chunks found in {path}")
    return chunks


def encode_chunks(
    chunks: list[dict[str, Any]], model_name: str, batch_size: int
) -> np.ndarray:
    """Encode and L2-normalize chunk text for cosine similarity."""
    device = choose_device()
    print(f"Loading embedding model {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an inner-product index over normalized embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional array")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_metadata(chunks: list[dict[str, Any]], path: Path) -> None:
    """Save chunk metadata in the same order as vectors in the FAISS index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for index construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks_path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index_path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata_path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Encode chunks, build the FAISS index, and save retrieval artifacts."""
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be greater than 0")

    chunks = load_chunks(args.chunks_path)
    embeddings = encode_chunks(chunks, args.model_name, args.batch_size)
    index = build_index(embeddings)

    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_path))
    save_metadata(chunks, args.metadata_path)

    print(f"Saved FAISS index with {index.ntotal} vectors to {args.index_path}")
    print(f"Saved {len(chunks)} chunk metadata records to {args.metadata_path}")


if __name__ == "__main__":
    main()
