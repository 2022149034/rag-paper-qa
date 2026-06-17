"""Streamlit demo for scientific-paper direct and RAG question answering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from .answer import (
        MISTRAL_MODEL,
        QWEN_MODEL,
        generate_answer,
        generate_answer_with_model,
        load_model,
    )
    from .retrieve import DEFAULT_MODEL_NAME as EMBEDDING_MODEL
    from .retrieve import Retriever
except ImportError:
    from answer import (
        MISTRAL_MODEL,
        QWEN_MODEL,
        generate_answer,
        generate_answer_with_model,
        load_model,
    )
    from retrieve import DEFAULT_MODEL_NAME as EMBEDDING_MODEL
    from retrieve import Retriever


PAPERS_PATH = Path("data/qasper_sample.json")
INDEX_PATH = Path("data/faiss.index")
METADATA_PATH = Path("data/chunk_metadata.json")

MODEL_OPTIONS = {
    "Qwen": QWEN_MODEL,
    "Mistral": MISTRAL_MODEL,
}


@st.cache_data(show_spinner=False)
def load_papers(path: str) -> list[dict[str, Any]]:
    """Load paper records for the sidebar selector."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [dict(paper) for paper in data if isinstance(paper, Mapping)]


@st.cache_resource(show_spinner="Loading FAISS index and BGE-M3...")
def load_retriever(index_path: str, metadata_path: str) -> Retriever:
    """Cache the FAISS index, metadata, and embedding model."""
    return Retriever(index_path=Path(index_path), metadata_path=Path(metadata_path))


@st.cache_resource(show_spinner="Loading selected language model...")
def load_llm(model_name: str) -> tuple[Any, Any]:
    """Cache each selected tokenizer and LLM for repeated questions."""
    return load_model(model_name)


def index_is_ready() -> bool:
    """Return whether both required retrieval artifacts exist."""
    return INDEX_PATH.is_file() and METADATA_PATH.is_file()


def mock_evidence(paper: Mapping[str, Any], top_k: int) -> list[dict[str, Any]]:
    """Create lightweight evidence from paper text when mock retrieval is needed."""
    words = str(paper.get("paper_text", "")).split()
    if not words:
        return []

    evidence: list[dict[str, Any]] = []
    chunk_size = 160
    for index in range(min(top_k, (len(words) + chunk_size - 1) // chunk_size)):
        text = " ".join(words[index * chunk_size : (index + 1) * chunk_size])
        if text:
            evidence.append(
                {
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "chunk_id": f"mock_chunk_{index:03d}",
                    "score": max(0.0, 1.0 - index * 0.1),
                    "text": text,
                }
            )
    return evidence


def display_evidence(evidence: list[dict[str, Any]]) -> None:
    """Render retrieved chunks in expandable boxes with similarity scores."""
    st.subheader("Retrieved Supporting Evidence")
    if not evidence:
        st.info("No supporting evidence was retrieved.")
        return

    for index, chunk in enumerate(evidence, start=1):
        score = chunk.get("score")
        score_label = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"
        chunk_id = str(chunk.get("chunk_id", f"chunk_{index}"))
        with st.expander(f"Evidence {index}: {chunk_id} | Similarity: {score_label}"):
            st.write(chunk.get("text", ""))
            st.caption(
                f"Paper: {chunk.get('title', '')} | Paper ID: {chunk.get('paper_id', '')}"
            )


def render_info(
    model_label: str,
    model_name: str,
    mode: str,
    top_k: int,
    mock: bool,
    evidence_count: int,
) -> None:
    """Render model and retrieval configuration details."""
    st.subheader("Model and Retrieval Info")
    st.json(
        {
            "model": model_label,
            "model_name": model_name,
            "mode": mode,
            "embedding_model": EMBEDDING_MODEL if mode == "rag" else "Not used",
            "top_k": top_k if mode == "rag" else "Not used",
            "retrieved_chunks": evidence_count,
            "mock_mode": mock,
        }
    )


def main() -> None:
    """Render the Streamlit paper QA demo."""
    st.set_page_config(
        page_title="Scientific Paper Question Answering Assistant",
        layout="wide",
    )
    st.title("Scientific Paper Question Answering Assistant")
    st.caption("Compare direct LLM answers with evidence-grounded RAG answers.")

    try:
        papers = load_papers(str(PAPERS_PATH))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not load papers: {error}")
        papers = []

    if not papers:
        st.warning(
            "No papers are available. Run `python src/load_data.py` for real-paper "
            "questions. Mock mode can still be used to test the interface."
        )
        papers = [
            {
                "paper_id": "mock-paper",
                "title": "Mock paper placeholder",
                "paper_text": "",
            }
        ]

    paper_labels = [
        str(paper.get("title", "")).strip()
        or str(paper.get("paper_id", "Untitled paper"))
        for paper in papers
    ]
    with st.sidebar:
        st.header("Settings")
        selected_index = st.selectbox(
            "Select paper title",
            options=range(len(papers)),
            format_func=lambda index: paper_labels[index],
        )
        model_label = st.selectbox("Select model", options=list(MODEL_OPTIONS))
        mode_label = st.radio("Select mode", options=("Direct", "RAG"))
        top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
        mock = st.checkbox("Mock mode", value=False)

        if index_is_ready():
            st.success("FAISS index is ready.")
        else:
            st.warning("FAISS index has not been built.")

    selected_paper = papers[selected_index]
    question = st.text_area(
        "Question",
        placeholder="What is the main contribution of this paper?",
        height=120,
    )
    run = st.button("Run", type="primary", use_container_width=True)

    if not run:
        return
    if not question.strip():
        st.error("Enter a question before clicking Run.")
        return

    mode = mode_label.lower()
    model_name = MODEL_OPTIONS[model_label]
    evidence: list[dict[str, Any]] = []

    try:
        if mode == "rag":
            if index_is_ready():
                retriever = load_retriever(str(INDEX_PATH), str(METADATA_PATH))
                evidence = retriever.retrieve(
                    question,
                    paper_id=str(selected_paper.get("paper_id", "")) or None,
                    top_k=top_k,
                )
            elif mock:
                evidence = mock_evidence(selected_paper, top_k)
                st.warning("Using mock evidence because the FAISS index is unavailable.")
            else:
                st.error(
                    "RAG mode requires `data/faiss.index` and "
                    "`data/chunk_metadata.json`. Run `python src/build_index.py` first."
                )
                return

        with st.spinner("Generating answer..."):
            if mock:
                answer = generate_answer(
                    model_name=model_name,
                    question=question,
                    evidence=evidence,
                    mode=mode,
                    mock=True,
                )
            else:
                tokenizer, model = load_llm(model_name)
                answer = generate_answer_with_model(
                    tokenizer=tokenizer,
                    model=model,
                    question=question,
                    evidence=evidence,
                    mode=mode,
                )
    except Exception as error:
        st.error(f"Could not complete the request: {error}")
        return

    st.subheader("Answer")
    st.write(answer)
    if mode == "rag":
        display_evidence(evidence)
    render_info(model_label, model_name, mode, top_k, mock, len(evidence))


if __name__ == "__main__":
    main()
