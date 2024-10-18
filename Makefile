# Detect the operating system
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif

# Project-specific variables
PARSING_DIR := utils/parsing/unstructured-api
PARSING_VENV := venv

STREAMLIT_PORT := 8501

# Load environment variables from .env file
include .env
export $(shell sed 's/=.*//' .env)

# Docker-related variables
DOCKER_PUSH := $(DOCKER_PUSH)

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
    HOME := $(HOME)
    MKDIR := mkdir -p
    RM := rm -rf
    FIND := find
    PARSING_VENV_ACTIVATE := $(PARSING_VENV)/bin/activate
endif

# Common variables
PYENV_ROOT := $(HOME)/.pyenv
PATH := $(PYENV_ROOT)/bin:$(PATH)

DEFAULT_PYTHON_VERSION := 3.11.3

VENV_PATH := .venv
PYTHON_VERSION_RANGE := ">=3.10,<3.13"

TEST_SUITE_VENV := .test_suite_venv
TEST_SUITE_REQUIREMENTS := tests/requirements.txt
BASE_REQUIREMENTS := base-requirements.txt

# Set default value for PARSING
PARSING ?= false

# Conditionally include start-parsing-service
ifeq ($(PARSING),true)
PARSING_SERVICE_TARGET := start-parsing-service
else
PARSING_SERVICE_TARGET :=
endif

# Default target
.PHONY: all
all: ensure-system-dependencies venv install $(PARSING_SERVICE_TARGET) post-process
	@echo "Setup complete."

# Create a virtual environment and install dependencies
.PHONY: venv-install
venv-install: 
	@make venv
	@make install

# Repl.it specific targets for kit installation
.PHONY: replit-kit
replit-kit:
	@if [ -z "$(KIT)" ]; then \
		echo "Error: KIT variable is not set. Usage: make replit-kit KIT=<kit_name> [RUN_COMMAND=<command>]"; \
		exit 1; \
	fi
	@echo "Setting up kit $(KIT) for Repl.it..."
	@if [ ! -d "$(KIT)" ]; then \
		echo "Error: Kit directory '$(KIT)' not found."; \
		exit 1; \
	fi
	@echo "Installing dependencies for kit $(KIT)..."
	@cd $(KIT) && \
	pip install --upgrade pip && \
	if [ -f "requirements.txt" ]; then \
		pip install -r requirements.txt --no-cache-dir; \
	else \
		echo "Warning: requirements.txt not found in $(KIT). Skipping kit-specific dependencies."; \
	fi
	@echo "Downgrading NLTK to version 3.8.1..."
	@pip install nltk==3.8.1 --no-cache-dir
	@echo "Downloading NLTK punkt resource..."
	@python -c "import nltk; nltk.download('punkt')"
	@echo "Kit $(KIT) setup complete."
	@if [ -n "$(RUN_COMMAND)" ]; then \
		echo "Running command: $(RUN_COMMAND)"; \
		cd $(KIT) && eval $(RUN_COMMAND); \
	else \
		echo "No run command specified. Setup complete."; \
	fi

# Update the existing replit target to include the new kit option
.PHONY: replit
replit:
	@if [ -n "$(KIT)" ]; then \
		make replit-kit KIT=$(KIT) RUN_COMMAND="$(RUN_COMMAND)"; \
		make post-process-replit; \
	else \
		make replit-install post-process-replit; \
	fi

.PHONY: replit-install
replit-install:
	@echo "Installing dependencies for Repl.it (skipping system dependencies)..."
	pip install --upgrade pip
	pip install -r $(BASE_REQUIREMENTS) --no-cache-dir

.PHONY: start-parsing-service-replit
start-parsing-service-replit: replit-setup-parsing-service
	@echo "Starting parsing service in the background..."
	@cd $(PARSING_DIR) && \
	make run-web-app > parsing_service.log 2>&1 & \
	echo $$! > parsing_service.pid
	@echo "Parsing service started. PID stored in $(PARSING_DIR)/parsing_service.pid"
	@echo "Use 'make parsing-log' to view the service log."

.PHONY: replit-setup-parsing-service
replit-setup-parsing-service:
	@echo "Setting up parsing service for Repl.it..."
	@echo "Current directory: $$(pwd)"
	@echo "PARSING_DIR: $(PARSING_DIR)"
	@if [ -d "$(PARSING_DIR)" ]; then \
		cd $(PARSING_DIR) && \
		echo "Changed to directory: $$(pwd)" && \
		if [ -f "Makefile" ]; then \
			echo "Running make install..." && \
			make install; \
		else \
			echo "Error: Makefile not found in $(PARSING_DIR)"; \
			exit 1; \
		fi; \
	else \
		echo "Error: Directory $(PARSING_DIR) not found"; \
		exit 1; \
	fi

.PHONY: post-process-replit
post-process-replit:
	@echo "Post-processing installation for Repl.it..."
	pip uninstall -y google-search-results
	pip install google-search-results==2.4.2

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
		echo "Poppler is already installed: $$(which pdftoppm)"; \
	fi
else
	@if ! command -v pdftoppm &> /dev/null; then \
		echo "Poppler not found. Installing Poppler..."; \
		sudo apt-get update && sudo apt-get install -y poppler-utils; \
	elif ! dpkg-query -W -f='${Status}' poppler-utils 2>/dev/null | grep -q "ok installed"; then \
		echo "Poppler not found. Installing Poppler..."; \
		sudo apt-get update && sudo apt-get install -y poppler-utils; \
	else \
		echo "Poppler is already installed: $$(which pdftoppm)"; \
	fi
endif

# Ensure libheif is installed
.PHONY: ensure-libheif
ensure-libheif:
ifeq ($(DETECTED_OS),Windows)
	@echo "libheif installation on Windows is not supported in this Makefile. Please install it manually."
else ifeq ($(DETECTED_OS),Darwin)
	@if ! brew list libheif &>/dev/null; then \
		echo "libheif not found. Installing libheif..."; \
		brew install libheif; \
	else \
		echo "libheif is already installed."; \
	fi
else
	@if ! dpkg -s libheif-dev &>/dev/null; then \
		echo "libheif not found. Installing libheif..."; \
		sudo apt-get update && sudo apt-get install -y libheif-dev; \
	else \
		echo "libheif is already installed."; \
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
		echo "pyenv not found. Installing pyenv..."; \
		if [ "$(DETECTED_OS)" = "Darwin" ]; then \
			brew install pyenv; \
		else \
			curl https://pyenv.run | bash; \
			echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> ~/.bashrc; \
			echo 'command -v pyenv >/dev/null || export PATH="$$PYENV_ROOT/bin:$$PATH"' >> ~/.bashrc; \
			echo 'eval "$$(pyenv init -)"' >> ~/.bashrc; \
			source ~/.bashrc; \
		fi; \
		export PATH="$(HOME)/.pyenv/bin:$$PATH"; \
		eval "$$(pyenv init -)"; \
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

# Create base virtual environment
.PHONY: create-base-venv
create-base-venv: ensure-pyenv
	@echo "Creating or updating base virtual environment..."
	@if [ ! -d $(VENV_PATH) ]; then \
		pyenv install -s $(DEFAULT_PYTHON_VERSION); \
		pyenv local $(DEFAULT_PYTHON_VERSION); \
		$(PYTHON) -m venv $(VENV_PATH); \
		. $(VENV_PATH)/bin/activate; \
		$(PIP) install --upgrade pip; \
		deactivate; \
	else \
		echo "Base virtual environment already exists."; \
	fi

# Create or use existing virtual environment
.PHONY: venv
venv: create-base-venv install-python-versions
	@echo "Checking for virtual environment..."
	@if [ ! -d $(VENV_PATH) ]; then \
		echo "Creating new virtual environment..."; \
		$(PYTHON) -m venv $(VENV_PATH); \
	else \
		echo "Using existing virtual environment."; \
	fi

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
install: ensure-qpdf ensure-system-dependencies ensure-libheif
	@echo "Installing dependencies..."
	@. $(VENV_PATH)/bin/activate && \
	$(PIP) install --upgrade pip && \
	$(PIP) install -r $(BASE_REQUIREMENTS) --no-cache-dir && \
	deactivate

# Post-process installation
.PHONY: post-process
post-process:
	@echo "Post-processing installation..."
	@. $(VENV_PATH)/bin/activate && \
	$(PIP) uninstall -y google-search-results && \
	$(PIP) install google-search-results==2.4.2 && \
	deactivate

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
	@cd $(PARSING_DIR) && \
	bash -c '. $(PARSING_VENV_ACTIVATE) && \
	make run-web-app > parsing_service.log 2>&1 & \
	echo $$! > parsing_service.pid && \
	deactivate' || true
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
	@echo "Building Docker image for platform linux/amd64..."
	@if [ "$(USE_CACHE)" = "true" ]; then \
		DOCKER_BUILDKIT=1 docker build \
			--platform linux/amd64 \
			--build-arg PROD_MODE=$(PROD_MODE) \
			--secret id=env,src=.env \
			--cache-from $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) \
			--build-arg BUILDKIT_INLINE_CACHE=1 \
			-t $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) . || \
		DOCKER_BUILDKIT=1 docker build \
			--platform linux/amd64 \
			--build-arg PROD_MODE=$(PROD_MODE) \
			--secret id=env,src=.env \
			--build-arg BUILDKIT_INLINE_CACHE=1 \
			-t $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) .; \
	else \
		DOCKER_BUILDKIT=1 docker build \
			--platform linux/amd64 \
			--build-arg PROD_MODE=$(PROD_MODE) \
			--secret id=env,src=.env \
			--no-cache \
			--build-arg BUILDKIT_INLINE_CACHE=1 \
			-t $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) .; \
	fi

.PHONY: docker-push
docker-push:
	@if [ "$(DOCKER_PUSH)" = "true" ]; then \
		echo "Pushing Docker image to registry..."; \
		docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG); \
	else \
		echo "Skipping Docker push (DOCKER_PUSH is not set to true)"; \
	fi

.PHONY: docker-build-push
docker-build-push: docker-build docker-push

.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	docker run -it --rm -p 8005:8005 -p $(STREAMLIT_PORT):8501 \
		--env-file .env \
		$(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: docker-shell
docker-shell:
	@echo "Opening a shell in the Docker container..."
	docker run -it --rm -p 8005:8005 -p $(STREAMLIT_PORT):8501 \
		--env-file .env \
		$(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) /bin/bash

.PHONY: docker-run-kit
docker-run-kit:
	@echo "Running specific kit in Docker container..."
	@if [ -z "$(KIT)" ]; then \
		echo "Error: KIT variable is not set. Usage: make docker-run-kit KIT=<kit_name> [COMMAND=<command>]"; \
		exit 1; \
	fi
	@if [ -z "$(COMMAND)" ]; then \
		docker run -d --name $(KIT)_container --rm -p 8005:8005 -p $(STREAMLIT_PORT):8501 \
			--env-file .env \
			$(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) /bin/bash -c \
			"cd $(KIT) && streamlit run streamlit/app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false"; \
	else \
		docker run -d --name $(KIT)_container --rm -p 8005:8005 -p $(STREAMLIT_PORT):8501 \
			--env-file .env \
			$(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) /bin/bash -c \
			"cd $(KIT) && $(COMMAND)"; \
	fi
	@echo "Container $(KIT)_container started in detached mode."

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
		$(PIP) install --upgrade pip && \
		$(PIP) install -r $(TEST_SUITE_REQUIREMENTS) --no-cache-dir && \
		deactivate

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

# Extract all command-line arguments except the target itself
get_args = $(filter-out $@,$(MAKECMDGOALS))

# Format code using Ruff
.PHONY: format
format:
	@echo "Formatting code using Ruff ..."
	@. $(VENV_PATH)/bin/activate && \
	ruff format $(or $(module), $(call get_args), .) && \
	deactivate

# Lint and type-check code using Ruff and MyPy
.PHONY: lint
lint:
	@echo "Linting and type-checking code using Ruff & MyPy ..."
	@. $(VENV_PATH)/bin/activate && \
	ruff check --fix $(or $(module), $(call get_args), .) && \
	ruff check --fix --select I $(or $(module), $(call get_args), .) && \
	mypy --explicit-package-bases $(or $(module), $(call get_args), .) && \
	deactivate

# Format, lint, and type-check code using Ruff and MyPy
.PHONY: format-lint
format-lint:
	@echo "Formatting, linting, and type-checking code using Ruff & MyPy ..."
	@. $(VENV_PATH)/bin/activate && \
	ruff format $(or $(module), $(call get_args), .) && \
	ruff check --fix $(or $(module), $(call get_args), .) && \
	ruff check --fix --select I $(or $(module), $(call get_args), .) && \
	mypy --explicit-package-bases $(or $(module), $(call get_args), .) && \
	deactivate

# Universal match for arguments passed directly
%:
	@:

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all                    : Set up main project, create or use venv, install dependencies, optionally start parsing service, and post-process"
	@echo "                          Set PARSING=true to start the parsing service."
	@echo "  replit                 : Set up project for Repl.it (skips pyenv check)"
	@echo "  ensure-system-dependencies : Ensure Poppler and Tesseract are installed"
	@echo "  ensure-poppler         : Install Poppler if not already installed"
	@echo "  ensure-tesseract       : Install Tesseract if not already installed"
	@echo "  ensure-pyenv           : Install pyenv if not already installed (not supported on Windows)"
	@echo "  install-python-versions: Install specific Python version ($(DEFAULT_PYTHON_VERSION)) (not supported on Windows)"
	@echo "  ensure-qpdf            : Install qpdf if not already installed (required for pikepdf)"
	@echo "  venv                   : Create or use existing virtual environment"
	@echo "  install                : Install dependencies using pip"
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
	@echo "  make format [args]     : Format code using Ruff"
	@echo "  make lint [args]       : Lint and type-check code using Ruff & MyPy"
	@echo "  make format-lint [args]: Format, lint, and type-check code using Ruff & MyPy"
