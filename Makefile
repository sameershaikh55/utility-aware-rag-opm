.PHONY: install test index eval-smoke eval ui clean

install:
	uv sync

test:
	uv run pytest -v

index:
	uv run python -m src.retriever

eval-smoke:
	uv run python -m src.eval --limit 6 --output results/smoke_eval.json

eval:
	uv run python -m src.eval --output results/baseline_eval.json

ui:
	uv run streamlit run src/app.py

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__ results/smoke_eval.json
