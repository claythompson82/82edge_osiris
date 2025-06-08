# Default to GPU environment. Override with 'make install ENV_TYPE=cpu'
ENV_TYPE ?= gpu

# Define PyTorch index URL based on environment type
ifeq ($(ENV_TYPE),gpu)
    PYTORCH_INDEX_URL := https://download.pytorch.org/whl/cu121
else
    PYTORCH_INDEX_URL := https://download.pytorch.org/whl/cpu
endif

.PHONY: install compile-reqs test

# Single command to set up the entire environment
install: compile-reqs
	@echo "Installing all dependencies for $(ENV_TYPE) environment..."
	pip install -r requirements.txt --extra-index-url $(PYTORCH_INDEX_URL)
	pip install -r requirements-dev.txt --extra-index-url $(PYTORCH_INDEX_URL)
	@echo "Installation complete."

# Command to re-compile the requirements files
compile-reqs:
	@echo "Compiling requirements with PyTorch index: $(PYTORCH_INDEX_URL)..."
	pip-compile requirements.in -o requirements.txt --extra-index-url $(PYTORCH_INDEX_URL) --resolver=backtracking
	pip-compile requirements-dev.in -o requirements-dev.txt --extra-index-url $(PYTORCH_INDEX_URL) --resolver=backtracking

# Command to run the test suite
test:
	pytest
