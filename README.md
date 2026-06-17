# RAG-based Scientific Paper Question Answering Assistant

## Project Overview

This project is an LLM-based application for answering questions about
scientific papers using retrieval-augmented generation (RAG). It downloads a
small sample of papers from QASPER, converts each paper into searchable chunks,
retrieves relevant supporting evidence, and generates answers with Qwen or
Mistral.

The project compares four settings:

1. Direct LLM with Qwen
2. RAG with Qwen
3. Direct LLM with Mistral
4. RAG with Mistral

## Motivation

Scientific papers are long, technical, and difficult to search manually. A
direct language model may answer from its general knowledge without grounding
its response in the selected paper. RAG first retrieves relevant passages from
the paper and provides them to the model as evidence, making answers easier to
verify and reducing unsupported responses.

## Features

- Load scientific paper text and QA annotations from QASPER
- Split papers into overlapping word-based chunks
- Build an embedding index with `BAAI/bge-m3`
- Retrieve top-k supporting evidence using FAISS cosine similarity
- Answer questions using Qwen2.5-7B-Instruct or Mistral-7B-Instruct-v0.3
- Compare Direct LLM and RAG answer generation
- Run quantitative and manual-score evaluation
- Display answers, evidence chunks, and similarity scores in Streamlit
- Use mock mode for lightweight local testing

## Models

| Purpose | Hugging Face Model |
| --- | --- |
| Answer generation | `Qwen/Qwen2.5-7B-Instruct` |
| Answer generation | `mistralai/Mistral-7B-Instruct-v0.3` |
| Chunk and query embeddings | `BAAI/bge-m3` |

The generation models require substantial memory. Real inference is intended
for a GPU virtual machine. Transformers uses automatic device placement and
dtype selection where possible.

## Dataset

The project uses the Hugging Face dataset
[`allenai/qasper`](https://huggingface.co/datasets/allenai/qasper). QASPER
contains information-seeking questions, answers, supporting evidence, and full
text from scientific papers.

By default, the loader selects 10 papers and up to 30 QA samples for a small,
reproducible course-project experiment.

## Project Structure

```text
rag-paper-qa/
|-- data/
|   |-- qasper_sample.json
|   |-- qa_samples.json
|   |-- chunks.json
|   |-- faiss.index                 # Generated, not committed
|   `-- chunk_metadata.json         # Generated, not committed
|-- src/
|   |-- load_data.py
|   |-- chunk_text.py
|   |-- build_index.py
|   |-- retrieve.py
|   |-- answer.py
|   |-- evaluate.py
|   `-- app.py
|-- results/
|   `-- comparison_results.csv
|-- models/                         # Model weights are not committed
|-- slides/
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- main.py
```

## Installation

Python 3.10 or newer is recommended.

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build when running the full models on a GPU
machine.

## Running the Pipeline

Run the commands from the repository root in order.

### 1. Download and Normalize QASPER

```bash
python src/load_data.py --num_papers 10 --max_questions 30
```

Outputs:

- `data/qasper_sample.json`
- `data/qa_samples.json`

### 2. Chunk Paper Text

```bash
python src/chunk_text.py
```

The default configuration creates 500-word chunks with 100-word overlap and
writes `data/chunks.json`.

### 3. Build the FAISS Index

```bash
python src/build_index.py
```

This encodes chunks with `BAAI/bge-m3` and writes:

- `data/faiss.index`
- `data/chunk_metadata.json`

### 4. Run Evaluation

```bash
python src/evaluate.py --max_samples 30
```

Evaluation writes and frequently checkpoints
`results/comparison_results.csv`.

### 5. Start the Streamlit Demo

```bash
streamlit run src/app.py
```

For a remotely accessible GPU VM:

```bash
streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501
```

## Mock Mode

Mock mode tests the pipeline and UI without downloading or loading the large
Qwen and Mistral models. It is useful for local development, validating CSV
generation, and testing the Streamlit interface on machines without a GPU.

Run mock evaluation:

```bash
python src/evaluate.py --max_samples 30 --mock
```

Run a mock answer:

```bash
python src/answer.py --question "What is the main contribution?" --mode direct --mock
```

For the Streamlit demo, enable **Mock mode** in the sidebar. In RAG mode, the
application uses mock evidence if the FAISS index has not been built.

## Evaluation

The evaluator compares:

- Direct LLM versus RAG
- Qwen versus Mistral

Each row in `results/comparison_results.csv` contains the question, gold
answer, four generated answers, retrieved evidence, and blank manual-scoring
columns.

### Average Answer Score

Manually score each generated answer:

- `0`: wrong
- `1`: partially correct
- `2`: correct

Average these scores for each model and mode to compare answer quality.

### Evidence Hit Rate

Evidence Hit Rate measures whether at least one retrieved chunk has substantial
lexical overlap with the gold supporting evidence or gold answer:

```text
Evidence Hit Rate = samples with evidence_hit = 1 / total samples
```

The implemented heuristic removes common stopwords and checks token overlap.

## Expected Submission Files

- `summary_slides.pptx`
- Implementation code in `src/`
- `README.md`
- `requirements.txt`
- `results/comparison_results.csv`
- `demo.mp4`

## Notes

- Do not commit downloaded Qwen, Mistral, or embedding-model weights.
- Do not commit generated FAISS indexes or large model-cache files.
- Configure the Hugging Face cache outside the repository, such as with
  `HF_HOME`, when running on a GPU VM.
- `data/faiss.index`, `data/chunk_metadata.json`, common model-weight formats,
  and local model directories are excluded by `.gitignore`.
- Mistral access may require accepting its Hugging Face repository terms and
  authenticating with a Hugging Face token.
