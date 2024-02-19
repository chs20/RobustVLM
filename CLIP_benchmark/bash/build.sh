#!/bin/bash
# gathers results from a results directory and builds a csv
# enter results dir in format /path/to/results/*
export PYTHONPATH="../":"${PYTHONPATH}"
set -e
echo "Enter path to results directory: "
read RES_DIR
echo "building results csv... ${RES_DIR}"
RND=${RANDOM}${RANDOM}
python -m clip_benchmark.cli build ${RES_DIR} --output "res${RND}.csv"
echo "reformatting csv..."
python reformat_csv.py res${RND}.csv