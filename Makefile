FORMATTER := poetry run black
LINTER := poetry run flake8
IMPORT_SORTER := poetry run isort
TYPE_CHECKER := poetry run mypy

SRC := run.py plot.py
PORT := 8000

# If this project is not ready to pass mypy, remove `type` below.
.PHONY: check
check: format lint type

.PHONY: ci
ci: format_check lint type

# Idiom found at https://www.gnu.org/software/make/manual/html_node/Force-Targets.html
FORCE:

.PHONY: format
format:
	$(FORMATTER) $(SRC)
	$(IMPORT_SORTER) $(SRC)

.PHONY: format_check
format_check:
	$(FORMATTER) $(SRC) --check --diff
	$(IMPORT_SORTER) $(SRC) --check --diff

.PHONY: lint
lint:
	$(LINTER) $(SRC)

.PHONY: type
type:
	$(TYPE_CHECKER) $(SRC)
