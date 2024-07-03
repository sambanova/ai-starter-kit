#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

if [ "$1" = "run-kit" ] && [ -n "$2" ]; then
    KIT_DIR="$2"  # The second argument is the kit directory
    shift 2  # Remove the first two arguments from the argument list
    cd "$KIT_DIR"  # Change to the specified kit directory
    exec "$@"  # Execute the remaining arguments as a command
else
    exec "$@"  # If not running a kit, just execute the given command
fi