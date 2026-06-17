"""Download a small QASPER sample and normalize it for the RAG pipeline.

QASPER contains nested paper sections, questions, and answer annotations. This
module converts those fields into two simple JSON files:

* ``qasper_sample.json``: paper records with their selected questions.
* ``qa_samples.json``: a flattened list containing the same questions.

The default sample is intentionally small enough for local development.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "allenai/qasper"
DEFAULT_OUTPUT_DIR = Path("data")


def _is_sequence(value: Any) -> bool:
    """Return whether a value is a non-string sequence."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _as_records(value: Any) -> list[dict[str, Any]]:
    """Convert a list of records or a column-oriented mapping into records.

    Hugging Face datasets may expose nested ``Sequence`` features either as a
    list of dictionaries or as a dictionary whose values are parallel lists.
    """
    if value is None:
        return []
    if isinstance(value, Mapping):
        sequence_lengths = [
            len(item) for item in value.values() if _is_sequence(item)
        ]
        if not sequence_lengths:
            return [dict(value)]

        record_count = max(sequence_lengths)
        records: list[dict[str, Any]] = []
        for index in range(record_count):
            record: dict[str, Any] = {}
            for key, item in value.items():
                if _is_sequence(item):
                    record[key] = item[index] if index < len(item) else None
                else:
                    record[key] = item
            records.append(record)
        return records
    if _is_sequence(value):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _text(value: Any) -> str:
    """Convert a possibly nested value into clean plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Mapping):
        return " ".join(part for part in (_text(item) for item in value.values()) if part)
    if _is_sequence(value):
        return " ".join(part for part in (_text(item) for item in value) if part)
    return str(value).strip()


def combine_paper_text(paper: Mapping[str, Any]) -> str:
    """Combine the abstract and full-text sections into one paper string."""
    parts: list[str] = []
    abstract = _text(paper.get("abstract"))
    if abstract:
        parts.append(f"Abstract\n{abstract}")

    full_text = paper.get("full_text", paper.get("paper_text"))
    if isinstance(full_text, str):
        cleaned = _text(full_text)
        if cleaned:
            parts.append(cleaned)
    else:
        for section in _as_records(full_text):
            section_name = _text(
                section.get("section_name", section.get("section", section.get("heading")))
            )
            paragraphs = _text(
                section.get(
                    "paragraphs",
                    section.get("paragraph", section.get("text", section.get("content"))),
                )
            )
            if paragraphs:
                parts.append(f"{section_name}\n{paragraphs}" if section_name else paragraphs)

    return "\n\n".join(parts)


def _normalize_answer(annotation: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    """Return a normalized gold answer and its evidence from one annotation."""
    answer = annotation.get("answer", annotation)
    if not isinstance(answer, Mapping):
        answer_text = _text(answer)
        return (answer_text, []) if answer_text else None

    evidence_value = answer.get(
        "evidence",
        answer.get(
            "supporting_evidence",
            answer.get("highlighted_evidence", annotation.get("evidence", [])),
        ),
    )
    if _is_sequence(evidence_value):
        evidence = [_text(item) for item in evidence_value]
    else:
        evidence_text = _text(evidence_value)
        evidence = [evidence_text] if evidence_text else []
    evidence = [item for item in evidence if item]

    free_form = _text(answer.get("free_form_answer", answer.get("free_form")))
    if free_form:
        return free_form, evidence

    yes_no = answer.get("yes_no")
    if _is_sequence(yes_no) and len(yes_no) == 1:
        yes_no = yes_no[0]
    if isinstance(yes_no, bool):
        return ("Yes" if yes_no else "No"), evidence
    yes_no_text = _text(yes_no)
    if yes_no_text:
        return yes_no_text.capitalize(), evidence

    extractive_spans = answer.get("extractive_spans", answer.get("extractive_span", []))
    spans = [_text(item) for item in extractive_spans] if _is_sequence(extractive_spans) else []
    spans = [item for item in spans if item]
    if spans:
        return "; ".join(spans), evidence

    unanswerable = answer.get("unanswerable")
    if _is_sequence(unanswerable) and len(unanswerable) == 1:
        unanswerable = unanswerable[0]
    if unanswerable is True:
        return "Unanswerable", evidence
    return None


def extract_qas(paper: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract questions with at least one valid answer annotation."""
    normalized_qas: list[dict[str, Any]] = []
    for index, qa in enumerate(_as_records(paper.get("qas", paper.get("questions")))):
        question = _text(qa.get("question", qa.get("query")))
        if not question:
            continue

        answer_value = qa.get("answers", qa.get("answer"))
        if isinstance(answer_value, Mapping) and any(
            key in answer_value
            for key in ("free_form_answer", "yes_no", "extractive_spans", "unanswerable")
        ):
            answer_records = [dict(answer_value)]
        else:
            answer_records = _as_records(answer_value)
        normalized_answer = next(
            (
                answer
                for annotation in answer_records
                if (answer := _normalize_answer(annotation)) is not None
            ),
            None,
        )
        if normalized_answer is None:
            continue

        gold_answer, supporting_evidence = normalized_answer
        question_id = _text(qa.get("question_id", qa.get("id"))) or str(index)
        normalized_qas.append(
            {
                "question_id": question_id,
                "question": question,
                "gold_answer": gold_answer,
                "supporting_evidence": supporting_evidence,
            }
        )
    return normalized_qas


def normalize_paper(paper: Mapping[str, Any], fallback_id: str) -> dict[str, Any] | None:
    """Normalize one paper, returning ``None`` when it has no usable QAs."""
    qas = extract_qas(paper)
    if not qas:
        return None

    return {
        "paper_id": _text(paper.get("paper_id", paper.get("id"))) or fallback_id,
        "title": _text(paper.get("title")),
        "paper_text": combine_paper_text(paper),
        "qas": qas,
    }


def select_questions(
    papers: list[dict[str, Any]], max_questions: int
) -> list[dict[str, Any]]:
    """Select questions round-robin so each sampled paper stays represented."""
    selected: list[dict[str, Any]] = []
    question_index = 0
    while len(selected) < max_questions:
        added_this_round = False
        for paper in papers:
            if question_index < len(paper["qas"]):
                selected.append(
                    {"paper_id": paper["paper_id"], **paper["qas"][question_index]}
                )
                added_this_round = True
                if len(selected) == max_questions:
                    break
        if not added_this_round:
            break
        question_index += 1
    return selected


def build_samples(
    num_papers: int, max_questions: int, split: str, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load QASPER and build normalized paper and flattened QA samples."""
    dataset = load_dataset(DATASET_NAME, split=split, trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed)

    papers: list[dict[str, Any]] = []
    for index, raw_paper in enumerate(dataset):
        paper = normalize_paper(raw_paper, fallback_id=f"{split}-{index}")
        if paper is not None:
            papers.append(paper)
        if len(papers) == num_papers:
            break

    selected_qas = select_questions(papers, max_questions)
    selected_keys = {
        (qa["paper_id"], qa["question_id"], qa["question"]) for qa in selected_qas
    }

    flat_qas: list[dict[str, Any]] = []
    used_sample_ids: set[str] = set()
    for paper in papers:
        retained_qas: list[dict[str, Any]] = []
        for qa_index, qa in enumerate(paper["qas"]):
            if (paper["paper_id"], qa["question_id"], qa["question"]) not in selected_keys:
                continue

            retained_qas.append(
                {
                    "question": qa["question"],
                    "gold_answer": qa["gold_answer"],
                    "supporting_evidence": qa["supporting_evidence"],
                }
            )
            base_id = qa["question_id"] or f"{paper['paper_id']}-{qa_index}"
            sample_id = base_id
            suffix = 2
            while sample_id in used_sample_ids:
                sample_id = f"{base_id}-{suffix}"
                suffix += 1
            used_sample_ids.add(sample_id)
            flat_qas.append(
                {
                    "sample_id": sample_id,
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "question": qa["question"],
                    "gold_answer": qa["gold_answer"],
                    "supporting_evidence": qa["supporting_evidence"],
                }
            )
        paper["qas"] = retained_qas

    return papers, flat_qas


def save_json(data: Any, path: Path) -> None:
    """Write JSON data with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for QASPER sampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_papers", type=int, default=10)
    parser.add_argument("--max_questions", type=int, default=30)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Download, normalize, and save the requested QASPER sample."""
    args = parse_args()
    if args.num_papers < 1 or args.max_questions < 1:
        raise ValueError("--num_papers and --max_questions must both be positive")

    papers, qa_samples = build_samples(
        num_papers=args.num_papers,
        max_questions=args.max_questions,
        split=args.split,
        seed=args.seed,
    )
    save_json(papers, args.output_dir / "qasper_sample.json")
    save_json(qa_samples, args.output_dir / "qa_samples.json")

    print(f"Saved {len(papers)} papers to {args.output_dir / 'qasper_sample.json'}")
    print(f"Saved {len(qa_samples)} QA samples to {args.output_dir / 'qa_samples.json'}")


if __name__ == "__main__":
    main()
