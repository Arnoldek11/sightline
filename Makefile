.PHONY: setup data reproduce-quick reproduce check test lint

setup:
	uv sync

data:
	bash scripts/download_data.sh

reproduce-quick:
	uv run python scripts/reproduce.py --quick

reproduce:
	uv run python scripts/reproduce.py --full

check:
	uv run python scripts/check_data_layer.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts
