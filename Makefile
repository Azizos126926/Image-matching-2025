PYTHON ?= python
PACKAGE = imc2025
CONFIG ?= configs/kaggle_offline.yaml
SUBMISSION ?= outputs/submission.csv

.PHONY: install lint test run score clean

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check src tests
	black --check src tests

test:
	pytest -q

run:
	$(PYTHON) -m $(PACKAGE) run --config $(CONFIG) --submission-path $(SUBMISSION)

score:
	$(PYTHON) -m $(PACKAGE) score --config $(CONFIG) --submission-path $(SUBMISSION)

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
