"""Generate direct or retrieval-augmented answers with an instruction-tuned LLM.

The default model is Qwen 2.5 7B Instruct. Mistral 7B Instruct v0.3 can be
selected with ``--model_name``. Only one model is cached at a time so switching
models does not keep both 7B models in GPU memory.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_MODEL_NAME = QWEN_MODEL

DIRECT_PROMPT = """You are a scientific paper question answering assistant.
Answer the following question as accurately as possible.

Question:
{question}

Answer:"""

RAG_PROMPT = """You are a scientific paper question answering assistant.
Answer the question based only on the provided evidence.
If the evidence is insufficient, say that the paper does not provide enough information.

Question:
{question}

Evidence:
{evidence}

Answer:"""

_LOADED_MODEL_NAME: str | None = None
_LOADED_TOKENIZER: Any = None
_LOADED_MODEL: Any = None


def format_evidence(retrieved_chunks: Sequence[Mapping[str, Any]]) -> str:
    """Format retrieved chunk dictionaries into a readable evidence block."""
    evidence_parts: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        labels: list[str] = []
        if chunk.get("title"):
            labels.append(str(chunk["title"]))
        if chunk.get("chunk_id"):
            labels.append(str(chunk["chunk_id"]))
        if isinstance(chunk.get("score"), (int, float)):
            labels.append(f"score={float(chunk['score']):.4f}")

        heading = " | ".join(labels)
        evidence_parts.append(
            f"[Evidence {index}{': ' + heading if heading else ''}]\n{text}"
        )
    return "\n\n".join(evidence_parts)


def build_prompt(question: str, evidence: Any = None, mode: str = "rag") -> str:
    """Build the requested direct or evidence-grounded prompt."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if mode not in {"direct", "rag"}:
        raise ValueError("mode must be either 'direct' or 'rag'")

    if mode == "direct":
        return DIRECT_PROMPT.format(question=question)

    if isinstance(evidence, str):
        evidence_text = evidence.strip()
    elif isinstance(evidence, Sequence):
        evidence_text = format_evidence(
            [chunk for chunk in evidence if isinstance(chunk, Mapping)]
        )
    elif evidence is None:
        evidence_text = ""
    else:
        raise TypeError("evidence must be a string, a sequence of chunks, or None")

    return RAG_PROMPT.format(
        question=question,
        evidence=evidence_text or "(No evidence was retrieved.)",
    )


def _release_loaded_model() -> None:
    """Release the currently cached model before loading a different one."""
    global _LOADED_MODEL_NAME, _LOADED_TOKENIZER, _LOADED_MODEL
    _LOADED_MODEL_NAME = None
    _LOADED_TOKENIZER = None
    _LOADED_MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(model_name: str) -> tuple[Any, Any]:
    """Load and cache one tokenizer/model pair using automatic device placement."""
    global _LOADED_MODEL_NAME, _LOADED_TOKENIZER, _LOADED_MODEL
    if _LOADED_MODEL_NAME == model_name:
        return _LOADED_TOKENIZER, _LOADED_MODEL

    _release_loaded_model()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} with device_map='auto' and torch_dtype='auto'")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _LOADED_MODEL_NAME = model_name
    _LOADED_TOKENIZER = tokenizer
    _LOADED_MODEL = model
    return tokenizer, model


def load_model(model_name: str) -> tuple[Any, Any]:
    """Load an LLM for callers that manage their own resource cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def _input_device(model: Any) -> torch.device:
    """Find the model device that should receive tokenized inputs."""
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _tokenize_prompt(tokenizer: Any, prompt: str) -> Mapping[str, torch.Tensor]:
    """Apply an instruction model's chat template, with a plain-text fallback."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    return tokenizer(prompt, return_tensors="pt")


def _mock_answer(question: str, evidence: Any, mode: str) -> str:
    """Return a deterministic lightweight response without loading an LLM."""
    if mode == "direct":
        return f"[Mock direct answer] A model would answer: {question.strip()}"

    evidence_text = evidence if isinstance(evidence, str) else ""
    if isinstance(evidence, Sequence) and not isinstance(evidence, str):
        evidence_text = format_evidence(
            [chunk for chunk in evidence if isinstance(chunk, Mapping)]
        )
    if not str(evidence_text).strip():
        return "The paper does not provide enough information."
    return "[Mock RAG answer] An answer would be generated from the provided evidence."


def generate_answer(
    model_name: str,
    question: str,
    evidence: Any = None,
    mode: str = "rag",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    mock: bool = False,
) -> str:
    """Generate an answer in direct or RAG mode.

    Set ``mock=True`` to test prompt and pipeline behavior without loading model
    weights. Normal generation uses sampling when ``temperature`` is positive
    and greedy decoding when it is zero.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be greater than 0")
    if temperature < 0:
        raise ValueError("temperature must be 0 or greater")

    prompt = build_prompt(question, evidence=evidence, mode=mode)
    if mock:
        return _mock_answer(question, evidence, mode)

    tokenizer, model = _load_model(model_name)
    inputs = {
        name: tensor.to(_input_device(model))
        for name, tensor in _tokenize_prompt(tokenizer, prompt).items()
    }
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_answer_with_model(
    tokenizer: Any,
    model: Any,
    question: str,
    evidence: Any = None,
    mode: str = "rag",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Generate an answer with an already-loaded tokenizer/model pair."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be greater than 0")
    if temperature < 0:
        raise ValueError("temperature must be 0 or greater")

    prompt = build_prompt(question, evidence=evidence, mode=mode)
    inputs = {
        name: tensor.to(_input_device(model))
        for name, tensor in _tokenize_prompt(tokenizer, prompt).items()
    }
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output_ids[0, prompt_length:], skip_special_tokens=True
    ).strip()


def load_evidence(path: Path) -> Any:
    """Load evidence from JSON, accepting a chunk list or a wrapped result list."""
    with path.open("r", encoding="utf-8") as file:
        evidence = json.load(file)
    if isinstance(evidence, Mapping):
        for key in ("chunks", "results", "evidence"):
            if key in evidence:
                return evidence[key]
    return evidence


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for answer generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=("direct", "rag"), default="rag")
    parser.add_argument("--evidence_path", type=Path)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Test without loading or downloading an LLM.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and print an answer from command-line arguments."""
    args = parse_args()
    evidence = load_evidence(args.evidence_path) if args.evidence_path else None
    answer = generate_answer(
        model_name=args.model_name,
        question=args.question,
        evidence=evidence,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        mock=args.mock,
    )
    print(answer)


if __name__ == "__main__":
    main()
