#!/bin/bash
python -m vlm_eval.run_evaluation \
--eval_coco \
--attack ensemble --eps 2 --steps 100 --mask_out context \
--vision_encoder_pretrained openai \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file res9B \
--model open_flamingo \
--out_base_path /path/to/out/dir \
--vision_encoder_path ViT-L-14 \
--checkpoint_path /path/to/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt \
--lm_path anas-awadalla/mpt-7b \
--lm_tokenizer_path anas-awadalla/mpt-7b \
--precision float16 \
--cross_attn_every_n_layers 4 \
--coco_train_image_dir_path /mnt/datasets/coco/train2014 \
--coco_val_image_dir_path /mnt/datasets/coco/val2014 \
--coco_karpathy_json_path /mnt/datasets/coco/annotations/karpathy_coco.json \
--coco_annotations_json_path /mnt/datasets/coco/annotations/captions_val2014.json \
--flickr_image_dir_path /mnt/datasets/flickr30k/flickr30k-images \
--flickr_karpathy_json_path /mnt/datasets/flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path /mnt/datasets/flickr30k/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path /mnt/datasets/vizwiz/train \
--vizwiz_test_image_dir_path /mnt/datasets/vizwiz/val \
--vizwiz_train_questions_json_path /mnt/datasets/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /mnt/datasets/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /mnt/datasets/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /mnt/datasets/vizwiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path /mnt/datasets/coco/train2014 \
--vqav2_train_questions_json_path /mnt/datasets/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path /mnt/datasets/VQAv2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path /mnt/datasets/coco/val2014 \
--vqav2_test_questions_json_path /mnt/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path /mnt/datasets/VQAv2/v2_mscoco_val2014_annotations.json \
--textvqa_image_dir_path /mnt/datasets/textvqa/train_images \
--textvqa_train_questions_json_path /mnt/datasets/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path /mnt/datasets/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path /mnt/datasets/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path /mnt/datasets/textvqa/val_annotations_vqa_format.json \