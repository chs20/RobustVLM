#!/bin/bash

python -m vlm_eval.run_evaluation \
--eval_coco \
--verbose \
--attack apgd --eps 2 --steps 10000 --mask_out none \
--targeted --target_str "Visit https://tinyurl.com/23cbjxjz" \
--vision_encoder_pretrained openai \
--dtype float32 \
--num_samples 25 \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--out_base_path /path/to/out/dir \
--model_path liuhaotian/llava-v1.5-7b \
--coco_train_image_dir_path /mnt/datasets/coco/train2014 \
--coco_val_image_dir_path /mnt/datasets/coco/val2014 \
--coco_karpathy_json_path /mnt/datasets/coco/annotations/karpathy_coco.json \
--coco_annotations_json_path /mnt/datasets/coco/annotations/captions_val2014.json
