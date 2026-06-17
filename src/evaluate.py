"""Evaluate direct and RAG answers across Qwen and Mistral.

The evaluator retrieves evidence once per QA sample, generates answers in
model-specific passes to avoid repeatedly swapping 7B models, and checkpoints
the comparison CSV after every sample update.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

try:
    from .answer import MISTRAL_MODEL, QWEN_MODEL, generate_answer
    from .retrieve import Retriever
except ImportError:
    from answer import MISTRAL_MODEL, QWEN_MODEL, generate_answer
    from retrieve import Retriever


DEFAULT_QA_PATH = Path("data/qa_samples.json")
DEFAULT_OUTPUT_PATH = Path("results/comparison_results.csv")
DEFAULT_INDEX_PATH = Path("data/faiss.index")
DEFAULT_METADATA_PATH = Path("data/chunk_metadata.json")

CSV_COLUMNS = [
    "sample_id",
    "paper_id",
    "title",
    "question",
    "gold_answer",
    "direct_qwen_answer",
    "rag_qwen_answer",
    "direct_mistral_answer",
    "rag_mistral_answer",
    "retrieved_evidence",
    "evidence_hit",
    "direct_qwen_score",
    "rag_qwen_score",
    "direct_mistral_score",
    "rag_mistral_score",
]

ANSWER_COLUMNS = (
    "direct_qwen_answer",
    "rag_qwen_answer",
    "direct_mistral_answer",
    "rag_mistral_answer",
)

SAMPLE_METADATA_COLUMNS = {
    "sample_id",
    "paper_id",
    "title",
    "question",
    "gold_answer",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def load_qa_samples(path: Path, max_samples: int) -> list[dict[str, Any]]:
    """Load and validate a small list of flattened QA samples."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(data[:max_samples]):
        if not isinstance(sample, Mapping):
            raise ValueError(f"QA sample {index} is not a JSON object")
        if not str(sample.get("question", "")).strip():
            continue
        samples.append(dict(sample))
    return samples


def lexical_tokens(text: Any) -> set[str]:
    """Return lowercase content-word tokens for a simple overlap heuristic."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def substantial_overlap(chunk_text: str, reference_text: str) -> bool:
    """Return whether a chunk substantially covers a reference answer/evidence."""
    chunk_tokens = lexical_tokens(chunk_text)
    reference_tokens = lexical_tokens(reference_text)
    if not chunk_tokens or not reference_tokens:
        return False

    overlap = len(chunk_tokens & reference_tokens)
    coverage = overlap / len(reference_tokens)
    required_overlap = 2
    return overlap >= required_overlap and coverage >= 0.35


def compute_evidence_hit(
    retrieved_chunks: Sequence[Mapping[str, Any]],
    supporting_evidence: Any,
    gold_answer: Any,
) -> int:
    """Score one when any retrieved chunk overlaps with gold evidence or answer."""
    references: list[str] = []
    if isinstance(supporting_evidence, str):
        references.append(supporting_evidence)
    elif isinstance(supporting_evidence, Sequence):
        references.extend(str(item) for item in supporting_evidence if item)
    if gold_answer:
        references.append(str(gold_answer))

    return int(
        any(
            substantial_overlap(str(chunk.get("text", "")), reference)
            for chunk in retrieved_chunks
            for reference in references
        )
    )


def new_result_row(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Create one output row with blank answer and manual-score columns."""
    return {
        "sample_id": sample.get("sample_id", ""),
        "paper_id": sample.get("paper_id", ""),
        "title": sample.get("title", ""),
        "question": sample.get("question", ""),
        "gold_answer": sample.get("gold_answer", ""),
        "direct_qwen_answer": "",
        "rag_qwen_answer": "",
        "direct_mistral_answer": "",
        "rag_mistral_answer": "",
        "retrieved_evidence": "[]",
        "evidence_hit": 0,
        "direct_qwen_score": "",
        "rag_qwen_score": "",
        "direct_mistral_score": "",
        "rag_mistral_score": "",
    }


def save_results(rows: list[dict[str, Any]], path: Path) -> None:
    """Atomically checkpoint all current results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    """Load an existing comparison CSV, ignoring extra columns if present."""
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            {column: row.get(column, "") for column in CSV_COLUMNS}
            for row in reader
        ]


def sample_key(sample: Mapping[str, Any]) -> str:
    """Return a stable key for matching samples across checkpoints."""
    sample_id = str(sample.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return f"{sample.get('paper_id', '')}\n{sample.get('question', '')}"


def is_mock_answer(value: Any) -> bool:
    """Return whether a cell contains one of this project's mock answers."""
    return str(value).lstrip().startswith("[Mock ")


def prepare_result_rows(
    samples: list[dict[str, Any]],
    output_path: Path,
    mock: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    """Create rows, preserving resumable checkpoint values when available."""
    existing_rows = load_existing_results(output_path)
    existing_by_key = {
        sample_key(row): row
        for row in existing_rows
        if sample_key(row)
    }

    reused_rows = 0
    cleared_mock_answers = 0
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = new_result_row(sample)
        existing_row = existing_by_key.get(sample_key(sample))
        if existing_row:
            reused_rows += 1
            for column in CSV_COLUMNS:
                if column not in SAMPLE_METADATA_COLUMNS:
                    row[column] = existing_row.get(column, row[column])

            if not mock:
                for column in ANSWER_COLUMNS:
                    if is_mock_answer(row.get(column, "")):
                        row[column] = ""
                        cleared_mock_answers += 1

        rows.append(row)

    return rows, reused_rows, cleared_mock_answers


def needs_answer_generation(row: Mapping[str, Any], column: str, mock: bool) -> bool:
    """Return whether an answer cell should be generated in this run."""
    answer = str(row.get(column, "")).strip()
    if not answer:
        return True
    return not mock and is_mock_answer(answer)


def parse_cached_evidence(row: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Return checkpointed evidence when it is present and well formed."""
    raw_evidence = str(row.get("retrieved_evidence", "")).strip()
    if not raw_evidence or raw_evidence == "[]":
        return None

    try:
        evidence = json.loads(raw_evidence)
    except json.JSONDecodeError:
        return None

    if not isinstance(evidence, list) or not all(
        isinstance(chunk, Mapping) for chunk in evidence
    ):
        return None
    return [dict(chunk) for chunk in evidence]


def load_cached_evidence(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]] | None:
    """Return all checkpointed evidence, or None when any row needs retrieval."""
    cached_evidence = [parse_cached_evidence(row) for row in rows]
    if any(evidence is None for evidence in cached_evidence):
        return None
    return [evidence for evidence in cached_evidence if evidence is not None]


def retrieve_all(
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    retriever: Retriever,
    top_k: int,
    output_path: Path,
) -> list[list[dict[str, Any]]]:
    """Return evidence for every sample, retrieving only missing rows."""
    all_evidence: list[list[dict[str, Any]]] = []
    for sample, row in tqdm(
        zip(samples, rows), total=len(samples), desc="Retrieving evidence"
    ):
        cached_evidence = parse_cached_evidence(row)
        if cached_evidence is not None:
            all_evidence.append(cached_evidence)
            continue

        paper_id = str(sample.get("paper_id", "")).strip() or None
        evidence = retriever.retrieve(
            str(sample["question"]), paper_id=paper_id, top_k=top_k
        )
        all_evidence.append(evidence)
        row["retrieved_evidence"] = json.dumps(evidence, ensure_ascii=False)
        row["evidence_hit"] = compute_evidence_hit(
            evidence,
            sample.get("supporting_evidence", []),
            sample.get("gold_answer", ""),
        )
        save_results(rows, output_path)
    return all_evidence


def generate_model_answers(
    model_name: str,
    model_label: str,
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    all_evidence: list[list[dict[str, Any]]],
    output_path: Path,
    mock: bool,
) -> None:
    """Generate direct and RAG answers for one model, checkpointing each sample."""
    for sample, row, evidence in tqdm(
        zip(samples, rows, all_evidence),
        total=len(samples),
        desc=f"Generating {model_label} answers",
    ):
        question = str(sample["question"])
        direct_column = f"direct_{model_label}_answer"
        rag_column = f"rag_{model_label}_answer"

        if needs_answer_generation(row, direct_column, mock):
            row[direct_column] = generate_answer(
                model_name=model_name,
                question=question,
                mode="direct",
                mock=mock,
            )
            save_results(rows, output_path)

        if needs_answer_generation(row, rag_column, mock):
            row[rag_column] = generate_answer(
                model_name=model_name,
                question=question,
                evidence=evidence,
                mode="rag",
                mock=mock,
            )
            save_results(rows, output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the four-setting evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa_path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--index_path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata_path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate mock answers without loading Qwen or Mistral.",
    )
    return parser.parse_args()


def main() -> None:
    """Run retrieval, four answer settings, and checkpoint the comparison CSV."""
    args = parse_args()
    if args.top_k < 1 or args.max_samples < 1:
        raise ValueError("--top_k and --max_samples must both be greater than 0")

    samples = load_qa_samples(args.qa_path, args.max_samples)
    if not samples:
        raise ValueError(f"No usable QA samples found in {args.qa_path}")

    rows, reused_rows, cleared_mock_answers = prepare_result_rows(
        samples, args.output_path, args.mock
    )
    if reused_rows:
        print(f"Loaded {reused_rows} rows from existing checkpoint {args.output_path}")
    if cleared_mock_answers:
        print(
            "Cleared "
            f"{cleared_mock_answers} mock answer cells so real generation can resume"
        )

    cached_evidence = load_cached_evidence(rows)
    if cached_evidence is None:
        retriever = Retriever(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
        )
        all_evidence = retrieve_all(
            samples, rows, retriever, args.top_k, args.output_path
        )
        del retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        all_evidence = cached_evidence
        print(f"Reused retrieved evidence from checkpoint for {len(rows)} rows")

    generate_model_answers(
        QWEN_MODEL, "qwen", samples, rows, all_evidence, args.output_path, args.mock
    )
    generate_model_answers(
        MISTRAL_MODEL,
        "mistral",
        samples,
        rows,
        all_evidence,
        args.output_path,
        args.mock,
    )

    hit_rate = sum(int(row["evidence_hit"]) for row in rows) / len(rows)
    print(f"Saved {len(rows)} evaluation rows to {args.output_path}")
    print(f"Evidence Hit Rate: {hit_rate:.2%}")


if __name__ == "__main__":
    main()
