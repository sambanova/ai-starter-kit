# Contributing to SambaNova AI Starter Kits

Thank you for your interest in contributing to SambaNova AI Starter Kits! This document provides guidelines and best practices for contributing to our repository.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Branch Protection Rules](#branch-protection-rules)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Package Owner Responsibilities](#package-owner-responsibilities)

## Overview

SambaNova AI Starter Kits are open-source examples and guides designed to facilitate the deployment of AI-driven use cases. We welcome contributions that improve existing kits or add new functionality.

## Getting Started

1. Create a branch (not a fork) for your contribution
2. Set up your development environment [See Here For Instructions](https://github.com/sambanova/ai-starter-kit?tab=readme-ov-file#2-base-environment-setup)
3. Make your changes following our coding standards
4. Submit a pull request

## Branch Protection Rules

The `main` branch is protected with the following rules:

- Pull requests must be up-to-date with the base branch
- All conversations must be resolved before merging
- At least one reviewer approval is required
- All GitHub Actions checks must pass

## Development Workflow

### Branch Naming Conventions

Use the following prefixes for your branches:

- `feature/` for new features (e.g., `feature/rag-component`)
- `improvement/` for improvements (e.g., `improvement/add-langchain-cache`)
- `bugfix/` for bug fixes (e.g., `bugfix/vectorstore`)
- `documentation/` for documentation (e.g., `documentation/rag-component`)
- `release/` for releases (e.g., `release/v1.0.1`)

### Code Quality Standards

#### Format and Lint Requirements

All code must pass our automated checks using Ruff and MyPy. You can run these checks locally using:

```bash
# Using Makefile (Preferred)
make venv-install  # Initialize environment
make format       # Format code with Ruff
make lint         # Lint and type check with Ruff & MyPy
make format-lint  # Run all checks

# Direct commands (Optional if you've ran the make commands above)
ruff format your_module
ruff check --fix your_module
ruff check --fix --select I your_module
mypy --explicit-package-bases your_module
```

#### Testing Requirements

- All contributions must pass the unit test suite
- Tests are automatically run when opening a PR
- Package owners must maintain up-to-date tests with good coverage
- New AISK Modules ie Kits (at high effort level with UI, etc.) should have unit test coverage that has been reviewed in a tests folder within the module. See the enterprise_knowledge_retreiver kit for an example.

## Pull Request Guidelines

### Opening a PR

1. Use the appropriate prefix in your PR title:
   - `Feature:` for new features
   - `Improvement:` for improvements
   - `Bugfix:` for bug fixes
   - `Documentation:` for documentation
   - `Release:` for releases

2. Include:
   - Informative title following the above conventions
   - Detailed description
   - Appropriate label
   - Self-assignment

### Before Merging

Ensure:
- Branch is up-to-date with main
- All conversations are resolved
- At least one reviewer has approved
- All checks have passed
- Code is formatted and linted
- All tests pass

### After Merging

- Delete your branch

## Package Owner Responsibilities

Package owners must:

1. Maintain up-to-date unit tests with good coverage
2. Implement both 'main' and 'github_pull_request' test suites
3. Clear all deprecation warnings
4. Update libraries monthly and coordinate with global dependency updates

## Virtual Environment

- Create the virtual environment as `.venv` at the repo root
- This ensures consistency with common practices and Makefile configuration

## Dependencies

- List dependencies alphabetically in `requirements.txt` for the kit 
- Add any new dependencies to `base-requirements.txt` in the root folder (if not already present)


## Questions or Issues?
- <a href="https://community.sambanova.ai/latest" target="_blank">,Message us</a> on SambaNova Community <a href="https://community.sambanova.ai/latest" 
- Create an issue on GitHub
- We're happy to help!

---

Note: These contribution guidelines are subject to change. Always refer to the latest version in the repository.