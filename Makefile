PYTHON ?= python3

.PHONY: test check-provenance lint

test:
	$(PYTHON) -m pytest tests/ -v

check-provenance:
	$(PYTHON) scripts/check_provenance.py

lint:
	ruff check .
	mypy .
