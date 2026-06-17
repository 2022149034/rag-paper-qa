# Final Project Summary

## 1. Project Goal

This project builds a retrieval-augmented question answering system for scientific papers. The goal is to compare direct large language model answering with retrieval-grounded answering, and to evaluate whether retrieved evidence improves factual accuracy on paper-specific questions.

The system compares two instruction-tuned 7B models:

- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct-v0.3

It also compares two answer settings:

- Direct LLM answering, where the model receives only the question.
- RAG answering, where the model receives retrieved evidence from the paper.

## 2. RAG Pipeline

The implemented pipeline is:

1. Load QASPER scientific paper data.
2. Convert paper questions, answers, and evidence into flattened QA samples.
3. Split paper text into retrievable chunks.
4. Embed chunks with BGE-M3.
5. Store chunk embeddings in a FAISS vector index.
6. Embed each user question at query time.
7. Retrieve the top-k most relevant chunks.
8. Insert retrieved evidence into a RAG prompt.
9. Generate answers with Qwen or Mistral.
10. Save direct and RAG outputs to `results/comparison_results.csv`.

The main implementation files are in `src/`:

- `load_data.py`
- `chunk_text.py`
- `build_index.py`
- `retrieve.py`
- `answer.py`
- `evaluate.py`
- `app.py`

## 3. QASPER Data Processing

The project uses QASPER-style scientific paper question answering data. The data is converted into a flattened sample format containing:

- paper id
- paper title
- question
- gold answer
- supporting evidence

The final evaluation used `data/qa_samples.json`. Although the evaluation command used `--max_samples 30`, the current dataset contains 22 usable QA samples, so the final results file contains 22 rows.

## 4. BGE-M3 + FAISS Retrieval

The retriever uses:

- embedding model: `BAAI/bge-m3`
- vector search backend: FAISS
- default retrieval depth: `top_k=5`

Each retrieved evidence item includes:

- paper id
- title
- chunk id
- similarity score
- chunk text

The FAISS index and chunk metadata are generated artifacts and should not be committed to GitHub. They can be rebuilt from the data and code.

## 5. Qwen vs Mistral

The project evaluates:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

On this evaluation set, Qwen performed best in the RAG setting. Both models struggled in direct mode because they often lacked paper-specific context and produced generic answers or asked for more information.

Average manual scores:

| Setting | Average Score |
|---|---:|
| Direct Qwen | 0.18 |
| RAG Qwen | 1.64 |
| Direct Mistral | 0.14 |
| RAG Mistral | 1.27 |

## 6. Direct LLM vs RAG

Direct LLM mode gives the model only the question. This is difficult for scientific paper QA because many questions ask about specific experiments, datasets, baselines, or results from a particular paper.

RAG mode first retrieves relevant paper chunks and then asks the model to answer using only that evidence. The results show a clear improvement:

- Qwen improved from 0.18 to 1.64 average score.
- Mistral improved from 0.14 to 1.27 average score.

This confirms that retrieval grounding is important for factual scientific paper QA.

## 7. Evidence Hit Rate

Evidence Hit Rate measures whether at least one retrieved chunk has substantial lexical overlap with the gold supporting evidence or gold answer.

Final result:

- total samples: 22
- evidence hits: 21
- Evidence Hit Rate: 95.45%

This indicates that BGE-M3 + FAISS usually retrieved relevant evidence for the current sample set. However, this metric evaluates retrieval overlap, not final answer correctness.

## 8. Manual Scoring Method

Each generated answer was manually scored using a 0/1/2 rubric:

- `0`: wrong, missing, or not aligned with the gold answer.
- `1`: partially correct, relevant but incomplete or imprecise.
- `2`: correct, covering the core gold-answer fact.

Scored columns in `results/comparison_results.csv`:

- `direct_qwen_score`
- `rag_qwen_score`
- `direct_mistral_score`
- `rag_mistral_score`

Score distribution:

| Setting | Score 0 | Score 1 | Score 2 | Average |
|---|---:|---:|---:|---:|
| Direct Qwen | 18 | 4 | 0 | 0.18 |
| RAG Qwen | 2 | 4 | 16 | 1.64 |
| Direct Mistral | 19 | 3 | 0 | 0.14 |
| RAG Mistral | 3 | 10 | 9 | 1.27 |

## 9. Streamlit Demo

The Streamlit demo in `src/app.py` provides an interactive interface for paper QA.

Demo features:

- load paper titles
- select a paper
- select Qwen or Mistral
- choose Direct or RAG mode
- set `top_k`
- enable mock mode for lightweight UI testing
- show whether the FAISS index is ready
- generate an answer
- display retrieved evidence in RAG mode
- display model and retrieval configuration

The demo was verified with:

- paper loading
- mock-mode UI testing
- FAISS index loading
- RAG evidence retrieval
- real Qwen generation

## 10. Current Limitations and Future Improvements

Current limitations:

- The final evaluation set is small, with 22 usable samples.
- Evidence Hit Rate is lexical and does not fully measure semantic relevance.
- Manual scoring was performed by one evaluator and may be subjective.
- Direct mode is disadvantaged because it receives no paper context.
- Retrieval uses dense search only, without reranking.
- Chunking is simple and may split important context.
- Answers do not include explicit evidence citations.
- Running real models requires a GPU and local Hugging Face model cache.

Future improvements:

- evaluate on more QASPER samples
- add a reranker after FAISS retrieval
- improve section-aware or paragraph-aware chunking
- generate answers with explicit citations
- add semantic or LLM-based evaluation
- compare additional model sizes and architectures
- improve the Streamlit UI with side-by-side answer comparison
- package reproducible build scripts for FAISS artifacts
