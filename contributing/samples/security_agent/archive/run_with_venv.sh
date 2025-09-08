#!/bin/bash

# Script to run Python files with the virtual environment activated

# Activate the virtual environment
source "$(dirname "$0")/venv/bin/activate"

# Run the Python script with all arguments passed to this script
python "$@"