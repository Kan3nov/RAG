*This activity has been created as part of the 42 curriculum by klafi*

# RAG against the machine

## Description
This project implements a Retrieval-Augmented Generation (RAG) pipeline for a large codebase. The goal is to answer natural-language questions by finding the most relevant source snippets, injecting them into the model context, and generating grounded answers instead of relying on the model's static training knowledge alone.

The repository is built around the `vllm` corpus stored under `data/raw`, which is indexed, retrieved, and scored against public evaluation datasets. The system is designed to follow the 42 project requirements: ingest and index a codebase, retrieve evidence with a lexical ranker, generate answers from that evidence, and evaluate retrieval quality with recall@k.

## System architecture
The pipeline is composed of four stages that mirror the project specification:

- Indexing: `src/indexing.py` walks the raw corpus, detects file types (`.py`, `.md`, `.txt`, `.sh`), and splits each file into reusable chunks. Metadata such as file path and character offsets is preserved so retrieval can be linked back to the exact source.
- Retrieval: `src/searching.py` loads the persisted BM25 index from `data/processed` and searches for the top-k matches given a user query.
- Answer generation: `src/answering.py` builds a prompt from the retrieved chunks and sends it to a local transformer model (`Qwen/Qwen3-0.6B`) through Hugging Face's `pipeline` API.
- Evaluation: `src/evaluation.py` compares retrieved chunks with the expected answer sources and computes recall based on file overlap and character span intersection.

The main entry point is `src/__main__.py`, which exposes the full CLI via Python Fire. The project also relies on Pydantic models defined in `src/models.py` to validate data exchanged between stages.

## Chunking strategy
The indexer uses a recursive chunking strategy via `langchain_text_splitters.RecursiveCharacterTextSplitter`:

- Python, Markdown, and shell files are split with language-aware settings.
- Plain text files are split using the default recursive splitter.
- Chunks are created with a configurable `max_chunk_size` and zero overlap, keeping each segment compact and highly relevant to the surrounding code or documentation.
- Each chunk stores its `file_path`, `content`, and character boundaries to enable precise evaluation and source attribution.

This allows the retriever to work on focused semantic units instead of whole files, which is essential for large codebases and for meeting recall requirements.

## Retrieval method
The retrieval stage uses BM25 via `bm25s` as the ranking function.

- The corpus is tokenized and indexed once during the indexing step.
- A query is tokenized in the same way and passed to the BM25 retriever.
- The system returns the highest-scoring chunks by relevance score.
- If the scores are too low, the system falls back to a placeholder result instead of producing noisy or misleading matches.

This is a lexical retrieval approach, which works well for exact terminology, function names, configuration keys, and documentation references that are frequently repeated in code and docs.

## Answer generation
The answer stage turns retrieval output into a grounded prompt:

- It concatenates the top retrieved chunks into a context window.
- It constructs a prompt using the Qwen instruction format.
- It checks the prompt length against the model's context limits and truncates the input if necessary.
- It invokes the local transformer model to generate a response based only on the provided evidence.

This keeps answers anchored to the retrieved corpus and reduces hallucination risk when the correct answer exists in the code or docs.

## Performance analysis
The project was designed to satisfy the project's evaluation constraints:

- Indexing time: under 5 minutes for the full corpus.
- Retrieval throughput: below 90 seconds for 200 questions.
- Recall@5: at least 80% on documentation questions and 50% on code questions.

The evaluator measures recall using a low-overlap criterion: a retrieved chunk counts as correct when it is in the same file and overlaps the reference span enough to pass the required threshold. This means retrieval quality depends on both file-level precision and approximate source alignment rather than exact span matching.

## Design decisions
Several implementation choices were made to keep the project robust and easy to run:

- Pydantic models are used to validate datasets and search results for consistent data flow.
- The CLI is built with Python Fire to keep the command surface simple and script-friendly.
- Files are stored under `data/raw`, `data/processed`, and `data/output` to keep raw data, generated indexes, and results separate.
- The project uses `uv` for dependency management and reproducible setup.
- Error handling is centralized in a utility helper to fail clearly with actionable messages when indexing or retrieval data is missing.

## Challenges faced
The main challenges in this project were the scale of the corpus and the need to balance precision with recall:

- Large repositories contain many similar identifiers and repeated patterns, which can cause noisy retrieval if chunking or tokenization is poorly configured.
- Code and documentation require different chunking behaviors, so language-aware splitting was necessary.
- The generated answer can be weak if the model receives too much irrelevant context, which is why retrieval is limited to the top-k matches.
- Evaluation requires a careful interpretation of source overlap, not exact text equality, so the project keeps exact file and offset metadata during indexing.

The solution was to combine a compact but structured chunking approach with a strong lexical ranker and explicit source metadata tracking.

## Instructions
### Prerequisites
- Python 3.10 or later
- `uv` installed on the system
- Enough disk space for the dataset and model dependencies

### Installation
```bash
uv sync
```

### Common commands
```bash
# Index the corpus
make index

# Search a single query
make search

# Search a dataset of questions
make search_dataset

# Generate an answer for a single question
make answer

# Generate answers for a full dataset
make answer_dataset

# Evaluate retrieval quality
make evaluate
```

### Direct CLI usage
```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search --query "How do I configure the OpenAI server?" -k 5
uv run python -m src search_dataset --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json -k 5 --save_dir data/output/search_results/UnansweredQuestions/
uv run python -m src answer --query "How do I configure the OpenAI server?" --k 5
uv run python -m src evaluate --std_search_res_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path datasets_public/public/AnsweredQuestions/dataset_code_public.json
```

## Example usage
```bash
# Build the index
make index

# Run retrieval on a whole public dataset
make search_dataset

# Generate answers for the retrieved evidence
make answer_dataset
```

The generated results are stored under `data/output`, while the public evaluation dataset lives under `datasets_public/public`.

## Repository layout
```text
.
├── data/
│   ├── raw/                     # source codebase to index
│   ├── processed/               # persisted BM25 index and chunk metadata
│   └── output/                  # retrieval and answer JSON outputs
├── datasets_public/
│   └── public/
│       ├── UnansweredQuestions/
│       └── AnsweredQuestions/
├── src/
│   ├── __main__.py              # Fire-based CLI entry point
│   ├── indexing.py              # chunking and index persistence
│   ├── searching.py             # BM25 retrieval logic
│   ├── answering.py             # prompt generation and answer inference
│   ├── evaluation.py            # recall@k evaluation logic
│   ├── models.py                # Pydantic models
│   ├── config.py                # project constants
│   └── utils.py                 # utility helpers
├── Makefile
├── pyproject.toml
├── uv.lock
├── README.md
└── moulinette/
```

## Resources
- Intro to RAG
  - https://youtu.be/T-D1OfcDW1M?si=w5ZgW0APIKYS28s2
- RAG In-Depth
  - https://youtu.be/swvzKSOEluc
- Different Chunking Techniques
  - https://youtu.be/Pk2BeaGbcTE
- Text Splitting Functions
  - https://docs.langchain.com/oss/python/integrations/splitters/code_splitter
- Python Fire
  - https://github.com/google/python-fire/blob/master/docs/guide.md
- BM25 vs. TF-IDF
  - https://medium.com/@jinmochong/what-is-bm25-comparison-with-tf-idf-and-beyond-5a740479214b
- BM25s
  - https://huggingface.co/blog/xhluca/bm25s
- tqdm
  - https://youtu.be/n4E7of9BINo
- Model Response Generation
  - https://youtu.be/1h6lfzJ0wZw

## AI usage
AI was used as a support tool for documentation tasks, structuring explanations, generating docstrings, and helping with README drafting. There was no AI-generated implementation at all.
