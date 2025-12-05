#!/bin/bash

# Load environment variables from .env file
set -a
source .env
set +a

# Activate virtual environment
source venv/bin/activate

# Run the benchmark
python run_meddiff_benchmark.py --models gpt-4 gpt-4o claude-sonnet-4-5-20250929 claude-3-5-sonnet-20241022 "$@"
