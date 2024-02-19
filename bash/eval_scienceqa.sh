#!/bin/bash

baseDir=/path/to/baseDir
baseModel='LLAVA'
# baseModel='openFlamingo'

modelPath=${1}
if [ -z "${modelPath}" ]
then
      echo "\$modelPath is empty Using robust model from here: "
      modelPath=/path/to/ckpt.pt
      modelPath1=ckpt_name
else
      echo "\$modelPath is NOT empty"
      modelPath1=${modelPath}
fi

answerFile="${baseModel}_${modelPath1}"
echo "Will save to the following json: "
echo $answerFile

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.5-7b \
    --eval-model ${baseModel} \
    --pretrained_rob_path ${modelPath} \
    --question-file "${baseDir}/llava_test_CQM-A.json" \
    --image-folder PATH-TO-scienceQA/test \
    --answers-file ${baseDir}/answers/${answerFile}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ${baseDir} \
    --result-file ${baseDir}/answers/${answerFile}.jsonl \
    --output-file ${baseDir}/answers/${answerFile}_output.jsonl \
    --output-result ${baseDir}/answers/${answerFile}_result.json
