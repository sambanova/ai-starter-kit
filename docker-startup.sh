#!/bin/bash
# Start the parsing service
cd /app/utils/parsing/unstructured-api
make run-web-app &

# Navigate to the specific kit directory and execute commands
cd /app
exec "$@"