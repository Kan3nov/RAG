run: install 
	@uv run python -m src

install:
	@uv sync

debug: install
	@uv run python -m pdb src/__main__.py

clean:
	@find \( -name "__pycache__" -o -name ".mypy_cache" \) -prune -exec rm -rf {} \;
	@rm -rf data

lint:
	@flake8 src/ && mypy src/ --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=skip