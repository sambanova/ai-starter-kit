# Makefile for AI Starter Kit project setup and management using Poetry and pip

# Variables
PYTHON := python3
POETRY := poetry
VENV_PATH := .venv
PYTHON_VERSION_RANGE := ">=3.10,<3.13"
REQUIREMENTS_FILE := base-requirements.txt

# Parsing service specific variables
PARSING_PYTHON_VERSION := 3.11.3
PYENV_ROOT := $(HOME)/.pyenv
PATH := $(PYENV_ROOT)/bin:$(PATH)
PARSING_PYTHON := $(PYENV_ROOT)/versions/$(PARSING_PYTHON_VERSION)/bin/python

# Project-specific variables
EKR_DIR := enterprise_knowledge_retriever
PARSING_DIR := utils/parsing/unstructured-api
PARSING_VENV := venv
EKR_COMMAND := streamlit run streamlit/app.py --browser.gatherUsageStats false

# Default target
.PHONY: all
all: venv install add-dependencies

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

# Install specific Python version using pyenv
.PHONY: install-python-version
install-python-version: ensure-pyenv
	@if [ ! -d $(PYENV_ROOT)/versions/$(PARSING_PYTHON_VERSION) ]; then \
		echo "Installing Python $(PARSING_PYTHON_VERSION)..."; \
		pyenv install $(PARSING_PYTHON_VERSION); \
	else \
		echo "Python $(PARSING_PYTHON_VERSION) is already installed."; \
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
venv: ensure-poetry init-poetry
	@echo "Checking for virtual environment..."
	@if [ ! -d $(VENV_PATH) ]; then \
		echo "Creating new virtual environment..."; \
		$(POETRY) config virtualenvs.in-project true; \
		$(POETRY) env use $(PYTHON); \
	else \
		echo "Using existing virtual environment."; \
	fi

# Install dependencies
.PHONY: install
install: venv
	@echo "Installing dependencies..."
	@$(POETRY) install --no-root

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
ekr: start-parsing-service
	@echo "Setting up Enterprise Knowledge Retriever project..."
	@cd $(EKR_DIR) && ( \
		if [ ! -d venv ]; then \
			echo "Creating new virtual environment for EKR..."; \
			$(PYTHON) -m venv venv; \
		else \
			echo "Using existing virtual environment for EKR."; \
		fi; \
		source venv/bin/activate && \
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
setup-parsing-service: install-python-version
	@echo "Setting up parsing service..."
	@cd $(PARSING_DIR) && ( \
		if [ ! -d $(PARSING_VENV) ]; then \
			echo "Creating new virtual environment for parsing service..."; \
			$(PARSING_PYTHON) -m venv $(PARSING_VENV); \
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

# Template for adding new subprojects using pip
define subproject_template
.PHONY: $(1)
$(1):
	@echo "Setting up $(1) project..."
	@cd $(2) && ( \
		if [ ! -d venv ]; then \
			echo "Creating new virtual environment for $(1)..."; \
			$$(PYTHON) -m venv venv; \
		else \
			echo "Using existing virtual environment for $(1)."; \
		fi; \
		source venv/bin/activate; \
		pip install --upgrade pip; \
		if [ -f requirements.txt ]; then \
			pip install -r requirements.txt; \
		else \
			echo "requirements.txt not found in $(2). Skipping dependency installation."; \
		fi; \
		deactivate; \
	)
endef

# Example of how to add a new subproject
# $(eval $(call subproject_template,subproject_name,subproject_directory))

# Clean up
.PHONY: clean
clean: stop-parsing-service
	@echo "Cleaning up..."
	@rm -rf $(VENV_PATH)
	@rm -rf $(PARSING_DIR)/$(PARSING_VENV)
	@rm -rf $(EKR_DIR)/venv
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete

# Format code using black
.PHONY: format
format:
	@echo "Formatting code..."
	@$(POETRY) run black .

# Run the main application (placeholder)
.PHONY: run
run:
	@echo "Running the application..."
	@$(POETRY) run python main.py

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all                    : Set up main project, create or use venv, install dependencies, and add from $(REQUIREMENTS_FILE)"
	@echo "  ensure-pyenv           : Install pyenv if not already installed"
	@echo "  install-python-version : Install specific Python version for parsing service"
	@echo "  ensure-poetry          : Install Poetry if not already installed"
	@echo "  init-poetry            : Initialize Poetry project if not already initialized"
	@echo "  venv                   : Create or use existing virtual environment"
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
	@echo "  run                    : Run the main application"
	@echo "  help                   : Show this help message"