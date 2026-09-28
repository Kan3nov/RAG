vpath %.py src

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
	@flake8 src/ json_schema/ && mypy src/ json_schema/ --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=skip