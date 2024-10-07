#!/usr/bin/env bash

# Check if mypy is installed
if ! command -v mypy &> /dev/null; then
    echo "Error: mypy is not installed or not in the system's PATH."
    exit 1
fi

# Check if the configuration file exists
CONFIG_FILE="./mypy.ini"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: The configuration file '$CONFIG_FILE' does not exist."
    exit 1
fi

# Run mypy with the configuration file and explicit package bases
if ! mypy --config-file="$CONFIG_FILE" --explicit-package-bases .; then
    echo "Error: mypy command failed."
    exit 1
fi
