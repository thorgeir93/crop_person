install:
   uv sync

run-workflow *args:
   uv run python -m src.crop_person.__main__ run-workflow {{args}}

check:
    uv run ruff format . --check
    uv run mypy .

fix:
   uv run ruff format .
