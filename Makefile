# Makefile for AI Starter Kit project setup and management using Poetry and pip

# Variables
PYENV_ROOT := $(HOME)/.pyenv
PATH := $(PYENV_ROOT)/bin:$(PATH)

DEFAULT_PYTHON_VERSION := 3.11.3
EKR_PYTHON_VERSION := 3.9.4

PYTHON := $(PYENV_ROOT)/versions/$(DEFAULT_PYTHON_VERSION)/bin/python
EKR_PYTHON := $(PYENV_ROOT)/versions/$(EKR_PYTHON_VERSION)/bin/python

POETRY := poetry
VENV_PATH := .venv
PYTHON_VERSION_RANGE := ">=3.10,<3.13"
REQUIREMENTS_FILE := base-requirements.txt

# Project-specific variables
EKR_DIR := enterprise_knowledge_retriever
PARSING_DIR := utils/parsing/unstructured-api
PARSING_VENV := venv
EKR_VENV := venv
EKR_COMMAND := streamlit run streamlit/app.py --browser.gatherUsageStats false

# Default target
.PHONY: all
all: venv update-lock validate install add-dependencies

# Install pyenv if not already installed
.PHONY: ensure-pyenv
ensure-pyenv:
	@if ! command -v pyenv &> /dev/null; then \
		echo "pyenv not found. Installing pyenv..."; \
		curl https://pyenv.run | bash; \
		echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> ~/.bashrc; \
		echo 'command -v pyenv >/dev/null || export PATH="$$PYENV_ROOT/bin:$$PATH"' >> ~/.bashrc; \
		echo 'eval "$$(pyenv init -)"' >> ~/.bashrc; \
		source ~/.bashrc; \
	fi

# Install specific Python versions using pyenv
.PHONY: install-python-versions
install-python-versions: ensure-pyenv
	@if [ ! -d $(PYENV_ROOT)/versions/$(DEFAULT_PYTHON_VERSION) ]; then \
		echo "Installing Python $(DEFAULT_PYTHON_VERSION)..."; \
		pyenv install $(DEFAULT_PYTHON_VERSION); \
	else \
		echo "Python $(DEFAULT_PYTHON_VERSION) is already installed."; \
	fi
	@if [ ! -d $(PYENV_ROOT)/versions/$(EKR_PYTHON_VERSION) ]; then \
		echo "Installing Python $(EKR_PYTHON_VERSION)..."; \
		pyenv install $(EKR_PYTHON_VERSION); \
	else \
		echo "Python $(EKR_PYTHON_VERSION) is already installed."; \
	fi

# Install Poetry if not already installed
.PHONY: ensure-poetry
ensure-poetry:
	@if ! command -v $(POETRY) &> /dev/null; then \
		echo "Poetry not found. Installing Poetry..."; \
		curl -sSL https://install.python-poetry.org | $(PYTHON) -; \
	fi

# Initialize Poetry project if pyproject.toml doesn't exist
.PHONY: init-poetry
init-poetry: ensure-poetry
	@if [ ! -f pyproject.toml ]; then \
		echo "Initializing Poetry project..."; \
		$(POETRY) init --no-interaction --python $(PYTHON_VERSION_RANGE); \
	fi

# Create or use existing virtual environment
.PHONY: venv
venv: ensure-poetry init-poetry install-python-versions
	@echo "Checking for virtual environment..."
	@if [ ! -d $(VENV_PATH) ]; then \
		echo "Creating new virtual environment..."; \
		$(POETRY) config virtualenvs.in-project true; \
		$(POETRY) env use $(PYTHON); \
	else \
		echo "Using existing virtual environment."; \
	fi

# Update lock file
.PHONY: update-lock
update-lock:
	@echo "Updating poetry.lock file..."
	@if [ -f poetry.lock ]; then \
		$(POETRY) lock --no-update; \
	else \
		$(POETRY) lock; \
	fi

# Validate project setup
.PHONY: validate
validate: update-lock
	@echo "Validating project setup..."
	@$(POETRY) check

# Install dependencies
.PHONY: install
install: update-lock
	@echo "Installing dependencies..."
	@$(POETRY) install --no-root --sync

# Add dependencies from base-requirements.txt
.PHONY: add-dependencies
add-dependencies: ensure-poetry
	@echo "Adding dependencies from $(REQUIREMENTS_FILE)..."
	@if [ -f $(REQUIREMENTS_FILE) ]; then \
		while read -r line; do \
			if [[ $$line != \#* && -n $$line ]]; then \
				$(POETRY) add $$line || echo "Failed to add: $$line"; \
			fi \
		done < $(REQUIREMENTS_FILE); \
	else \
		echo "$(REQUIREMENTS_FILE) not found. Skipping dependency addition."; \
	fi

# Set up Enterprise Knowledge Retriever project using pip and run the app
.PHONY: ekr
ekr: start-parsing-service install-python-versions
	@echo "Setting up Enterprise Knowledge Retriever project..."
	@cd $(EKR_DIR) && ( \
		if [ ! -d $(EKR_VENV) ]; then \
			echo "Creating new virtual environment for EKR using Python $(EKR_PYTHON_VERSION)..."; \
			$(EKR_PYTHON) -m venv $(EKR_VENV); \
		else \
			echo "Using existing virtual environment for EKR."; \
		fi; \
		. $(EKR_VENV)/bin/activate && \
		pip install --upgrade pip && \
		if [ -f requirements.txt ]; then \
			pip install -r requirements.txt; \
		else \
			echo "requirements.txt not found in $(EKR_DIR). Skipping dependency installation."; \
		fi && \
		echo "Starting EKR application..." && \
		$(EKR_COMMAND); \
	)

# Set up parsing service
.PHONY: setup-parsing-service
setup-parsing-service: install-python-versions
	@echo "Setting up parsing service..."
	@cd $(PARSING_DIR) && ( \
		if [ ! -d $(PARSING_VENV) ]; then \
			echo "Creating new virtual environment for parsing service..."; \
			$(PYTHON) -m venv $(PARSING_VENV); \
		else \
			echo "Using existing virtual environment for parsing service."; \
		fi; \
		. $(PARSING_VENV)/bin/activate && \
		pip install --upgrade pip && \
		make install && \
		deactivate || true; \
	)

# Start parsing service in the background
.PHONY: start-parsing-service
start-parsing-service: setup-parsing-service
	@echo "Starting parsing service in the background..."
	@cd $(PARSING_DIR) && ( \
		. $(PARSING_VENV)/bin/activate && \
		nohup make run-web-app > parsing_service.log 2>&1 & \
		echo $$! > parsing_service.pid && \
		deactivate || true; \
	)
	@echo "Parsing service started. PID stored in $(PARSING_DIR)/parsing_service.pid"
	@echo "Use 'make parsing-log' to view the service log."

# Stop parsing service
.PHONY: stop-parsing-service
stop-parsing-service:
	@echo "Stopping parsing service..."
	@PID=$$(lsof -ti tcp:8005); \
	if [ -n "$$PID" ]; then \
		kill -9 $$PID && \
		echo "Parsing service stopped (PID: $$PID)."; \
	else \
		echo "No parsing service found running on port 8005."; \
	fi
	@rm -f $(PARSING_DIR)/parsing_service.pid

# View parsing service log
.PHONY: parsing-log
parsing-log:
	@if [ -f $(PARSING_DIR)/parsing_service.log ]; then \
		tail -f $(PARSING_DIR)/parsing_service.log; \
	else \
		echo "Parsing service log not found. Is the service running?"; \
	fi

# Check parsing service status
.PHONY: parsing-status
parsing-status:
	@if [ -f $(PARSING_DIR)/parsing_service.pid ]; then \
		PID=$$(cat $(PARSING_DIR)/parsing_service.pid); \
		if ps -p $$PID > /dev/null; then \
			echo "Parsing service is running (PID: $$PID)"; \
		else \
			echo "Parsing service is not running (stale PID file found)"; \
			rm $(PARSING_DIR)/parsing_service.pid; \
		fi \
	else \
		echo "Parsing service is not running (no PID file found)"; \
	fi

# Clean up
.PHONY: clean
clean: stop-parsing-service
	@echo "Cleaning up..."
	@rm -rf $(VENV_PATH)
	@rm -rf $(PARSING_DIR)/$(PARSING_VENV)
	@rm -rf $(EKR_DIR)/$(EKR_VENV)
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete

# Format code using black
.PHONY: format
format:
	@echo "Formatting code..."
	@$(POETRY) run black .

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all                    : Set up main project, create or use venv, install dependencies, and add from $(REQUIREMENTS_FILE)"
	@echo "  ensure-pyenv           : Install pyenv if not already installed"
	@echo "  install-python-versions: Install specific Python versions ($(DEFAULT_PYTHON_VERSION) and $(EKR_PYTHON_VERSION))"
	@echo "  ensure-poetry          : Install Poetry if not already installed"
	@echo "  init-poetry            : Initialize Poetry project if not already initialized"
	@echo "  venv                   : Create or use existing virtual environment"
	@echo "  update-lock            : Update the poetry.lock file"
	@echo "  validate               : Validate the project setup"
	@echo "  install                : Install dependencies using Poetry (without installing the root project)"
	@echo "  add-dependencies       : Add dependencies from $(REQUIREMENTS_FILE) to Poetry"
	@echo "  ekr                    : Set up Enterprise Knowledge Retriever project, start parsing service, and run the EKR app"
	@echo "  setup-parsing-service  : Set up the parsing service environment"
	@echo "  start-parsing-service  : Start the parsing service in the background"
	@echo "  stop-parsing-service   : Stop the running parsing service"
	@echo "  parsing-log            : View the parsing service log in real-time"
	@echo "  parsing-status         : Check the status of the parsing service"
	@echo "  clean                  : Remove all virtual environments and cache files, stop parsing service"
	@echo "  format                 : Format code using black"
	@echo "  help                   : Show this help message"