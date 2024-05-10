#!/bin/bash

# Set default values
ENV_FILE="./.env"
CONFIG_FILE="config.yaml"

# Parse command line arguments
while getopts ":e:c:" opt; do
  case $opt in
    e) ENV_FILE="$OPTARG"
    ;;
    c) CONFIG_FILE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

# Check if the .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE file not found. Please make sure the .env file exists."
    exit 1
fi

# Load environment variables from the .env file
export $(grep -v '^#' "$ENV_FILE" | xargs)

# Check if UNSTRUCTURED_API_KEY is set
if [ -z "$UNSTRUCTURED_API_KEY" ]; then
    echo "Error: UNSTRUCTURED_API_KEY is not set in the .env file."
    exit 1
fi

# Check if the YAML configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE file not found. Please make sure the YAML configuration file exists."
    exit 1
fi

# Load the YAML configuration and extract the unstructured_port value
UNSTRUCTURED_PORT=$(awk '/unstructured_port:/ {print $2}' "$CONFIG_FILE")

# Set the default port if UNSTRUCTURED_PORT is not set
if [ -z "$UNSTRUCTURED_PORT" ]; then
    UNSTRUCTURED_PORT=8000
fi

# Run the Unstructured API container
docker run -p $UNSTRUCTURED_PORT:8000 -it --rm --name unstructured-api downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0 --env-file "$ENV_FILE" /bin/bash