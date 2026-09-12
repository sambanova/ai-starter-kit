#!/usr/bin/env bash
# Sourced (not executed) by vLLM/aiperf's quickstart scripts. Loads SAMBANOVA_API_BASE/
# SAMBANOVA_API_KEY from the repo-root .env if not already exported by the caller's shell -- an
# already-exported shell var always wins over .env. Uses python-dotenv (already a dependency)
# rather than `source .env`, since this repo's .env has "KEY = value" spacing plain bash can't
# parse.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    eval "$(python3 -c '
import shlex
from dotenv import dotenv_values
values = dotenv_values("'"$REPO_ROOT"'/.env")
for key in ("SAMBANOVA_API_BASE", "SAMBANOVA_API_KEY"):
    val = values.get(key) or ""
    print("DOTENV_" + key + "=" + shlex.quote(val))
')"
fi
: "${SAMBANOVA_API_BASE:=${DOTENV_SAMBANOVA_API_BASE:-}}"
: "${SAMBANOVA_API_KEY:=${DOTENV_SAMBANOVA_API_KEY:-}}"
