"""Split normalized scientific-paper text into overlapping word chunks.

The input is the paper-level JSON produced by ``load_data.py``. Each output
chunk retains its paper metadata so retrieval results can be traced back to the
source paper.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("data/qasper_sample.json")
DEFAULT_OUTPUT_PATH = Path("data/chunks.json")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping word windows, excluding empty chunks."""
    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def create_chunks(
    papers: list[Mapping[str, Any]], chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    """Create chunk records for every paper containing non-empty text."""
    chunks: list[dict[str, Any]] = []
    for paper in papers:
        paper_id = str(paper.get("paper_id", "")).strip()
        title = str(paper.get("title", "")).strip()
        paper_text = paper.get("paper_text", "")
        if not isinstance(paper_text, str):
            continue

        for chunk_index, text in enumerate(
            split_text(paper_text, chunk_size, chunk_overlap)
        ):
            chunks.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "chunk_id": f"{paper_id}_chunk_{chunk_index:03d}",
                    "chunk_index": chunk_index,
                    "text": text,
                }
            )
    return chunks


def load_papers(path: Path) -> list[Mapping[str, Any]]:
    """Load and validate the paper-level input JSON."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [paper for paper in data if isinstance(paper, Mapping)]


def save_chunks(chunks: list[dict[str, Any]], path: Path) -> None:
    """Save chunks as readable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for text chunking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate chunk sizing arguments before processing data."""
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be greater than 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk_overlap must be 0 or greater")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk_overlap must be smaller than --chunk_size")


def main() -> None:
    """Load papers, create overlapping chunks, and save them."""
    args = parse_args()
    validate_args(args)
    papers = load_papers(args.input_path)
    chunks = create_chunks(papers, args.chunk_size, args.chunk_overlap)
    save_chunks(chunks, args.output_path)
    print(f"Saved {len(chunks)} chunks to {args.output_path}")


if __name__ == "__main__":
    main()
