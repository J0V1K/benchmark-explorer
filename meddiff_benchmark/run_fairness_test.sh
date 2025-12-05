#!/bin/bash
set -a
source .env
set +a
source venv/bin/activate
python test_with_fairness_prompt.py "$@"
