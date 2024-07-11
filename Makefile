# Detect the operating system
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif

# Project-specific variables
PARSING_DIR := utils/parsing/unstructured-api
PARSING_VENV := venv

# Set OS-specific variables and commands
ifeq ($(DETECTED_OS),Windows)
    PYTHON := python
    PIP := pip
    HOME := $(USERPROFILE)
    MKDIR := mkdir
    RM := rmdir /s /q
    FIND := where
    PARSING_VENV_ACTIVATE := $(PARSING_DIR)\$(PARSING_VENV)\Scripts\activate.bat
else
    PYTHON := python3
    PIP := pip3
    MKDIR := mkdir -p
    RM := rm -rf
    FIND := find
    PARSING_VENV_ACTIVATE := $(PARSING_VENV)/bin/activate
endif

# Common variables
PYENV_ROOT := $(HOME)/.pyenv
PATH := $(PYENV_ROOT)/bin:$(PATH)

DEFAULT_PYTHON_VERSION := 3.11.3

POETRY := poetry
VENV_PATH := .venv
PYTHON_VERSION_RANGE := ">=3.10,<3.13"

TEST_SUITE_VENV := .test_suite_venv
TEST_SUITE_REQUIREMENTS := tests/requirements.txt


# Default target
.PHONY: all
all: 
	@make ensure-system-dependencies && \
	make venv && \
	make update-lock && \
	make validate && \
	make install && \
	make start-parsing-service && \
	make post-process || (echo "An error occurred during setup. Please check the output above." && exit 1)

# Ensure system dependencies (Poppler and Tesseract)
.PHONY: ensure-system-dependencies
ensure-system-dependencies: ensure-poppler ensure-tesseract

# Ensure Poppler is installed
.PHONY: ensure-poppler
ensure-poppler:
ifeq ($(DETECTED_OS),Windows)
	@where pdftoppm >nul 2>&1 || (echo Poppler not found. Please install it manually from https://github.com/oschwartz10612/poppler-windows/releases/ && exit 1)
else ifeq ($(DETECTED_OS),Darwin)
	@if ! command -v pdftoppm &> /dev/null; then \
		echo "Poppler not found. Installing Poppler..."; \
		brew install poppler; \
	else \
		echo "Poppler is already installed."; \
	fi
else
	@if ! command -v pdftoppm &> /dev/null; then \
		echo "Poppler not found. Installing Poppler..."; \
		sudo apt-get update && sudo apt-get install -y poppler-utils; \
	else \
		echo "Poppler is already installed: $$(which pdftoppm)"; \
	fi
endif

# Ensure Tesseract is installed
.PHONY: ensure-tesseract
ensure-tesseract:
ifeq ($(DETECTED_OS),Windows)
	@where tesseract >nul 2>&1 || (echo Tesseract not found. Please install it manually from https://github.com/UB-Mannheim/tesseract/wiki && exit 1)
else ifeq ($(DETECTED_OS),Darwin)
	@if ! command -v tesseract &> /dev/null; then \
		echo "Tesseract not found. Installing Tesseract..."; \
		brew install tesseract; \
	else \
		echo "Tesseract is already installed."; \
	fi
else
	@if ! command -v tesseract &> /dev/null; then \
		echo "Tesseract not found. Installing Tesseract..."; \
		sudo apt-get update && sudo apt-get install -y tesseract-ocr; \
	else \
		echo "Tesseract is already installed."; \
	fi
endif

# Ensure pyenv is available and set up
.PHONY: ensure-pyenv
ensure-pyenv:
ifeq ($(DETECTED_OS),Windows)
	@echo "pyenv is not supported on Windows. Please install Python $(DEFAULT_PYTHON_VERSION) manually."
else
	@if command -v pyenv &> /dev/null; then \
		echo "pyenv found. Setting up environment..."; \
		export PATH="$(HOME)/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
	else \
		echo "pyenv not found. Please install pyenv and run this script again."; \
		echo "You can install pyenv by following the instructions at: https://github.com/pyenv/pyenv#installation"; \
		exit 1; \
	fi
endif

# Install specific Python versions using pyenv
.PHONY: install-python-versions
install-python-versions: ensure-pyenv
ifeq ($(DETECTED_OS),Windows)
	@echo "Please ensure Python $(DEFAULT_PYTHON_VERSION) is installed manually on Windows."
else
	@if [ ! -d $(PYENV_ROOT)/versions/$(DEFAULT_PYTHON_VERSION) ]; then \
		echo "Installing Python $(DEFAULT_PYTHON_VERSION)..."; \
		pyenv install $(DEFAULT_PYTHON_VERSION); \
	else \
		echo "Python $(DEFAULT_PYTHON_VERSION) is already installed."; \
	fi
endif

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


# Create base virtual environment
.PHONY: create-base-venv
create-base-venv: ensure-pyenv
	@echo "Creating or updating base virtual environment..."
	@if [ ! -d $(VENV_PATH) ]; then \
		pyenv install -s $(DEFAULT_PYTHON_VERSION); \
		pyenv local $(DEFAULT_PYTHON_VERSION); \
		$(PYTHON) -m venv $(VENV_PATH); \
		. $(VENV_PATH)/bin/activate; \
		pip install --upgrade pip; \
		pip install poetry; \
		deactivate; \
	else \
		echo "Base virtual environment already exists."; \
	fi

# Create or use existing virtual environment
.PHONY: venv
venv: create-base-venv ensure-poetry init-poetry install-python-versions
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
	@$(POETRY) lock --no-update || (echo "Error updating lock file. Trying without --no-update flag..." && $(POETRY) lock)

# Validate project setup
.PHONY: validate
validate: update-lock
	@echo "Validating project setup..."
	@$(POETRY) check || (echo "Poetry check failed. Attempting to proceed." && true)


# Ensure qpdf is installed (for pikepdf)
.PHONY: ensure-qpdf
ensure-qpdf:
ifeq ($(DETECTED_OS),Windows)
	@where qpdf >nul 2>&1 || (echo qpdf not found. Please install it manually from https://github.com/qpdf/qpdf/releases)
else ifeq ($(DETECTED_OS),Darwin)
	@if ! command -v qpdf &> /dev/null; then \
		echo "qpdf not found. Installing qpdf..."; \
		brew install qpdf; \
	else \
		echo "qpdf is already installed."; \
	fi
else
	@if ! command -v qpdf &> /dev/null; then \
		echo "qpdf not found. Installing qpdf..."; \
		sudo apt-get update && sudo apt-get install -y qpdf; \
	else \
		echo "qpdf is already installed."; \
	fi
endif

# Install dependencies
.PHONY: install
install: update-lock ensure-qpdf ensure-system-dependencies
	@echo "Installing dependencies..."
	@$(POETRY) config virtualenvs.create false
	@$(POETRY) config installer.parallel false
	@$(POETRY) config experimental.new-installer false
	@$(POETRY) install --no-interaction --no-root --sync || \
	(echo "Initial install failed. Attempting to resolve conflicts..." && \
	$(POETRY) install --no-interaction --no-root --sync --remove-untracked)


# Post-process installation
.PHONY: post-process
post-process:
	@echo "Post-processing installation..."
	@$(POETRY) run pip uninstall -y google-search-results
	@$(POETRY) run pip install google-search-results==2.4.2

# Set up parsing service
.PHONY: setup-parsing-service
setup-parsing-service: install-python-versions
	@echo "Setting up parsing service..."
	@cd $(PARSING_DIR) && ( \
		echo "Current directory: $(shell pwd)"; \
		echo "PARSING_DIR: $(PARSING_DIR)"; \
		echo "PARSING_VENV: $(PARSING_VENV)"; \
		if [ ! -d $(PARSING_VENV) ]; then \
			echo "Creating new virtual environment for parsing service..."; \
			$(PYTHON) -m venv $(PARSING_VENV); \
		else \
			echo "Using existing virtual environment for parsing service."; \
		fi; \
		echo "Activating virtual environment: $(PARSING_VENV_ACTIVATE)"; \
		. $(PARSING_VENV_ACTIVATE) && \
		echo "Upgrading pip..."; \
		$(PIP) install --upgrade pip && \
		echo "Installing requirements..."; \
		make install && \
		echo "Deactivating virtual environment..."; \
		deactivate || true; \
	)

# Start parsing service in the background
.PHONY: start-parsing-service
start-parsing-service: setup-parsing-service
	@echo "Starting parsing service in the background..."
ifeq ($(DETECTED_OS),Windows)
	@cd $(PARSING_DIR) && ( \
		$(PARSING_VENV_ACTIVATE) && \
		start /b make run-web-app > parsing_service.log 2>&1 && \
		echo "Parsing service started. Check parsing_service.log for details." && \
		deactivate \
	)
else
	@cd $(PARSING_DIR) && ( \
		. $(PARSING_VENV_ACTIVATE) && \
		nohup make run-web-app > parsing_service.log 2>&1 & \
		echo $$! > parsing_service.pid && \
		deactivate || true; \
	)
	@echo "Parsing service started. PID stored in $(PARSING_DIR)/parsing_service.pid"
endif
	@echo "Use 'make parsing-log' to view the service log."

# Stop parsing service
.PHONY: stop-parsing-service
stop-parsing-service:
	@echo "Stopping parsing service..."
ifeq ($(DETECTED_OS),Windows)
	@for /f "tokens=5" %a in ('netstat -aon ^| find ":8005" ^| find "LISTENING"') do taskkill /F /PID %a
else
	@PID=$$(lsof -ti tcp:8005); \
	if [ -n "$$PID" ]; then \
		kill -9 $$PID && \
		echo "Parsing service stopped (PID: $$PID)."; \
	else \
		echo "No parsing service found running on port 8005."; \
	fi
	@rm -f $(PARSING_DIR)/parsing_service.pid
endif

# View parsing service log
.PHONY: parsing-log
parsing-log:
ifeq ($(DETECTED_OS),Windows)
	@if exist $(PARSING_DIR)\parsing_service.log (type $(PARSING_DIR)\parsing_service.log) else (echo Parsing service log not found. Is the service running?)
else
	@if [ -f $(PARSING_DIR)/parsing_service.log ]; then \
		tail -f $(PARSING_DIR)/parsing_service.log; \
	else \
		echo "Parsing service log not found. Is the service running?"; \
	fi
endif

# Check parsing service status
.PHONY: parsing-status
parsing-status:
ifeq ($(DETECTED_OS),Windows)
	@netstat -ano | findstr :8005 | findstr LISTENING > nul
	@if %errorlevel% equ 0 (echo Parsing service is running.) else (echo Parsing service is not running.)
else
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
endif

# Docker-related commands
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t ai-starter-kit .

.PHONY: docker-run
docker-run: docker-build
	@echo "Running Docker container..."
	docker run -it --rm -p 8005:8005 -p 8501:8501 ai-starter-kit

.PHONY: docker-shell
docker-shell: docker-build
	@echo "Opening a shell in the Docker container..."
	docker run -it --rm -p 8005:8005 -p 8501:8501 ai-starter-kit /bin/bash

.PHONY: docker-run-kit
docker-run-kit: docker-build
	@echo "Running specific kit in Docker container..."
	@if [ -z "$(KIT)" ]; then \
		echo "Error: KIT variable is not set. Usage: make docker-run-kit KIT=<kit_name> [COMMAND=<command>]"; \
		exit 1; \
	fi
	@if [ -z "$(COMMAND)" ]; then \
		docker run -it --rm -p 8005:8005 -p 8501:8501 ai-starter-kit /bin/bash -c "cd $(KIT) && streamlit run streamlit/app.py --browser.gatherUsageStats false"; \
	else \
		docker run -it --rm -p 8005:8005 -p 8501:8501 ai-starter-kit /bin/bash -c "cd $(KIT) && $(COMMAND)"; \
	fi

# Set up test suite
.PHONY: setup-test-suite
setup-test-suite: ensure-pyenv
	@echo "Setting up test suite environment..."
	@if [ ! -d $(PYENV_ROOT)/versions/$(DEFAULT_PYTHON_VERSION) ]; then \
		echo "Installing Python $(DEFAULT_PYTHON_VERSION) for test suite..."; \
		pyenv install $(DEFAULT_PYTHON_VERSION); \
	else \
		echo "Python $(DEFAULT_PYTHON_VERSION) is already installed."; \
	fi
	@pyenv local $(DEFAULT_PYTHON_VERSION)
	@$(PYTHON) -m venv $(TEST_SUITE_VENV)
	@. $(TEST_SUITE_VENV)/bin/activate && \
		pip install --upgrade pip && \
		pip install -r $(TEST_SUITE_REQUIREMENTS)

.PHONY: ensure-poetry-venv
ensure-poetry-venv:
	@if ! command -v poetry &> /dev/null; then \
		echo "Installing Poetry in test suite venv..."; \
		pip install poetry; \
	fi

.PHONY: clean-test-suite
clean-test-suite:
	@echo "Cleaning up test suite environment..."
	@rm -rf $(TEST_SUITE_VENV)
	@pyenv local --unset


# Clean up
.PHONY: clean
clean: stop-parsing-service 
	@echo "Cleaning up..."
ifeq ($(DETECTED_OS),Windows)
	@if exist $(VENV_PATH) rmdir /s /q $(VENV_PATH)
	@if exist $(PARSING_DIR)\$(PARSING_VENV) rmdir /s /q $(PARSING_DIR)\$(PARSING_VENV)
	@for /r %x in (*.pyc) do @del "%x"
	@for /d /r %x in (__pycache__) do @if exist "%x" rd /s /q "%x"
else
	@rm -rf $(VENV_PATH)
	@rm -rf $(PARSING_DIR)/$(PARSING_VENV)
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
endif

# Format code using black
.PHONY: format
format:
	@echo "Formatting code..."
	@$(POETRY) run black .

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all                    : Set up main project, create or use venv, install dependencies, start parsing service, and post-process"
	@echo "  ensure-system-dependencies : Ensure Poppler and Tesseract are installed"
	@echo "  ensure-poppler         : Install Poppler if not already installed"
	@echo "  ensure-tesseract       : Install Tesseract if not already installed"
	@echo "  ensure-pyenv           : Install pyenv if not already installed (not supported on Windows)"
	@echo "  install-python-versions: Install specific Python version ($(DEFAULT_PYTHON_VERSION)) (not supported on Windows)"
	@echo "  ensure-poetry          : Install Poetry if not already installed"
	@echo "  ensure-qpdf            : Install qpdf if not already installed (required for pikepdf)"
	@echo "  init-poetry            : Initialize Poetry project if not already initialized"
	@echo "  venv                   : Create or use existing virtual environment"
	@echo "  update-lock            : Update the poetry.lock file"
	@echo "  validate               : Validate the project setup"
	@echo "  install                : Install dependencies using Poetry (without installing the root project)"
	@echo "  post-process           : Perform post-installation steps (reinstall google-search-results)"
	@echo "  setup-parsing-service  : Set up the parsing service environment"
	@echo "  start-parsing-service  : Start the parsing service in the background"
	@echo "  stop-parsing-service   : Stop the running parsing service"
	@echo "  parsing-log            : View the parsing service log"
	@echo "  parsing-status         : Check the status of the parsing service"
	@echo "  docker-build           : Build Docker image"
	@echo "  docker-run             : Run Docker container"
	@echo "  docker-shell           : Open a shell in the Docker container"
	@echo "  docker-run-kit         : Run a specific kit in the Docker container. Usage: make docker-run-kit KIT=<kit_name> [COMMAND=<command>]"
	@echo "  setup-test-suite       : Set up the test suite environment"
	@echo "  clean-test-suite       : Clean up the test suite environment"
	@echo "  clean                  : Remove all virtual environments and cache files, stop parsing service"
	@echo "  format                 : Format code using black"
	@echo "  help                   : Show this help message"