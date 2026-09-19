PYTHON ?= python3

.PHONY: check test lint typecheck check-provenance

# Everything CI runs, in CI's order.
check: lint typecheck test check-provenance

test:
	$(PYTHON) -m pytest tests/ -v

# Policy (story 130.19): ruff's default rule set, configured in pyproject.toml exactly as
# in toke-corpus / toke-model. Invoked via `-m` so it works with a user-site install too.
lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy .

check-provenance:
	$(PYTHON) scripts/check_provenance.py
