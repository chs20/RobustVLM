#!/bin/bash

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

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --eval-model ${baseModel} \
    --pretrained_rob_path ${modelPath} \
    --question-file ./pope_eval/llava_pope_test.jsonl \
    --image-folder PATH_TO_COCO-VAL2014 \
    --answers-file ./pope_eval/${answerFile}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


python llava/eval/eval_pope.py \
    --model-name $answerFile \
    --annotation-dir ./pope_eval/coco/ \
    --question-file ./pope_eval/llava_pope_test.jsonl \
    --result-file ./pope_eval/${answerFile}.jsonl
