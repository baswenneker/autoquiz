# Makefile

# Variables
PYTHON = python3
PYTEST = pytest
SPHINX = sphinx-build
DOCS_DIR = docs
TESTS_DIR = tests

# Targets
.PHONY: test docs

test:
	$(PYTEST) $(TESTS_DIR)

docs:
	$(SPHINX) -b html $(DOCS_DIR) $(DOCS_DIR)/_build