# Multipart dataset upload - developer tasks.
# Ruff rules mirror applications/gateway (see pyproject.toml).

PYTHON ?= python3

# Dev tooling for make's own lint/test targets, installed on first use.
DEV_DEPS := ruff pytest

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: dev-install
dev-install: ## Install runtime deps + dev tooling (ruff, pytest)
	$(PYTHON) -m pip install -r requirements.txt $(DEV_DEPS)

.PHONY: lint
lint: ## Lint with ruff
	@$(PYTHON) -m ruff --version >/dev/null 2>&1 || $(PYTHON) -m pip install ruff
	$(PYTHON) -m ruff check .

.PHONY: format
format: ## Auto-format and apply safe lint fixes with ruff
	@$(PYTHON) -m ruff --version >/dev/null 2>&1 || $(PYTHON) -m pip install ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

.PHONY: format-check
format-check: ## Check formatting without modifying files (CI)
	@$(PYTHON) -m ruff --version >/dev/null 2>&1 || $(PYTHON) -m pip install ruff
	$(PYTHON) -m ruff format --check .

# `test` pulls the full dev env on demand (the suite imports the package, so it
# needs the runtime deps, not just pytest).
.PHONY: test
test: ## Run the test suite
	@$(PYTHON) -m pytest --version >/dev/null 2>&1 || $(MAKE) dev-install
	$(PYTHON) -m pytest

.PHONY: run
run: ## Run the interactive uploader
	$(PYTHON) run.py

# Deps for `make notebook` only, installed on first use.
JUPYTER_DEPS := jupytext jupyterlab ipykernel
# A uniquely-named kernel bound to THIS interpreter. Without it, `--set-kernel python3`
# picks whatever "python3" kernelspec Jupyter finds first (often a global/Homebrew one)
# which lacks this example's dependencies and can't import `src`, so the first cell fails.
NB_KERNEL := multipart-dataset-upload

.PHONY: notebook
notebook: install ## Open run.py as a Jupyter notebook, pinned to this env's kernel
	@$(PYTHON) -c "import jupytext, jupyterlab, ipykernel" 2>/dev/null || \
		{ echo "installing notebook deps ($(JUPYTER_DEPS))..."; $(PYTHON) -m pip install $(JUPYTER_DEPS); }
	$(PYTHON) -m ipykernel install --user --name $(NB_KERNEL) --display-name "Multipart dataset upload"
	$(PYTHON) -m jupytext --to ipynb --set-kernel $(NB_KERNEL) run.py
	$(PYTHON) -m jupyter lab run.ipynb; rm -f run.ipynb

.PHONY: clean
clean: ## Remove caches, local upload state, the generated notebook, and its kernel
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	@$(PYTHON) -m jupyter kernelspec remove -f $(NB_KERNEL) 2>/dev/null || true
	@rm -f .crusoe-upload-state.json run.ipynb
	@echo "clean"
