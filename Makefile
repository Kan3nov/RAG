vpath %.py src

SCRIPT := uv run python -m src

dataset_path ?= datasets_public/public/AnsweredQuestions/dataset_code_public.json
k ?= 5
save_dir ?= data/output/search_results/answered/
query ?= "How to configure the OpenAI server?"
max_chunk_size ?= 2000

all: run

run: install
	@uv run python -m src

install:
	@uv sync

debug: install
	@uv run python -m pdb src/__main__.py

clean:
	@find \( -name "__pycache__" -o -name ".mypy_cache" \) -prune -exec rm -rf {} \;
	@rm -rf data/output

lint:
	@flake8 src/ && mypy src/ --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=skip
index:
	$(SCRIPT) index --max_chunk_size $(max_chunk_size)

search:
	$(SCRIPT) search --query $(query) -k $(k)

search_dataset:
	$(SCRIPT) search_dataset --dataset_path $(dataset_path) -k $(k) -save_dir $(save_dir)
