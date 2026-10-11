vpath %.py src

SCRIPT := uv run python -m src
OP ?= index

dataset_path ?= data/datasets/UnansweredQuestions/dataset_code_public.json
answer_dataset_path ?= data/datasets/AnsweredQuestions/dataset_code_public.json

k ?= 5
search_save_dir ?= data/output/search_results/UnansweredQuestions/

query ?= "How to configure the OpenAI server?"
max_chunk_size ?= 2000

student_search_results_path ?= data/output/search_results/UnansweredQuestions/dataset_code_public.json
answer_save_dir ?= data/output/search_results_and_answer/UnansweredQuestions/

run: install index search_dataset
	./moulinette_pkg/moulinette-fedora evaluate_student_search_results \
	data/output/search_results/UnansweredQuestions/dataset_code_public.json \
	data/datasets/AnsweredQuestions/dataset_code_public.json --k 10

install:
	@uv sync

debug: install
	@uv run python -m pdb src/__main__.py $(OP)

clean:
	@find \( -name "__pycache__" -o -name ".mypy_cache" \) -prune -exec rm -rf {} \;
	@rm -rf data
	@rm -rf moulinette_pkg

lint:
	@flake8 src/ && mypy src/ --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=skip

prepare:
	tar -xf data+moulinette.tar.xz
	unzip datasets_public.zip
	unzip moulinette.zip
	unzip vllm-0.10.1.zip
	mkdir -p data/raw
	mv vllm-0.10.1 data/raw/
	mkdir -p data/datasets/
	mv public/* data/datasets/
	rm -rf public
	rm -rf vllm-0.10.1.zip
	rm -rf datasets_public.zip
	rm -rf moulinette.zip

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

evaluate:
	$(SCRIPT) evaluate --std_search_res_path $(student_search_results_path) --dataset_path $(answer_dataset_path)
