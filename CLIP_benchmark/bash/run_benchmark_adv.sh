#!/bin/bash
set -e # stop on error
SECONDS=0
# add parent dir to python path
export PYTHONPATH="../":"${PYTHONPATH}"
EPS=2  # TODO

SAMPLES=1000
SAVE_DIR=/path/to/out/dir  # TODO
mkdir -p "$SAVE_DIR"
python -m clip_benchmark.cli eval --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" --dataset benchmark/datasets.txt \
--pretrained_model benchmark/models.txt \
--output "${SAVE_DIR}/adv_{model}_{pretrained}_{dataset}_{n_samples}_bs{bs}_{attack}_{eps}_{iterations}.json" \
--attack aa --eps $EPS \
--batch_size 50 --n_samples $SAMPLES

hours=$((SECONDS / 3600))
minutes=$(( (SECONDS % 3600) / 60 ))
echo "[Runtime] $hours h $minutes min"