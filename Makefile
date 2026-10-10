vpath %.py src

SCRIPT := uv run python -m src

dataset_path ?= datasets_public/public/UnansweredQuestions/dataset_code_public.json
k ?= 5
search_save_dir ?= data/output/search_results/UnansweredQuestions/

query ?= "How to configure the OpenAI server?"
max_chunk_size ?= 2000

student_search_results_path ?= data/output/search_results/UnansweredQuestions/dataset_code_public.json
answer_save_dir ?= data/datasets/AnsweredQuestions/

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
	$(SCRIPT) search_dataset --dataset_path $(dataset_path) -k $(k) --save_dir $(search_save_dir)

answer:
	$(SCRIPT) answer --query $(query) --k $(k)

answer_dataset:
	$(SCRIPT) answer_dataset --std_search_res_path $(student_search_results_path) --save_dir $(answer_save_dir)