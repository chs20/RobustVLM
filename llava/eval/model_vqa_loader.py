import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import random
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from open_flamingo.eval.models.of_eval_model_adv import EvalModelAdv
from open_flamingo.eval.vqa_metric import (
    compute_vqa_accuracy,
    postprocess_vqa_generation,
)
from PIL import Image
import math

import warnings
warnings.filterwarnings("ignore")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_of_args(pretrained_rob_path=None):
    model_args = {}
    model_args['vision_encoder_pretrained'] = pretrained_rob_path
    model_args['vision_encoder_path'] =  'ViT-L-14'
    model_args['lm_path'] = 'anas-awadalla/mpt-7b'
    model_args['lm_tokenizer_path'] = 'anas-awadalla/mpt-7b'
    model_args['checkpoint_path'] = '/data/naman_deep_singh/project_multimodal/OpenFlamingo-9B-vitl-mpt7b.pt'
    # model_args['device'] = 'cuda'
    model_args['cross_attn_every_n_layers'] =  4 
    model_args['precision'] = 'float32'

    return model_args

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, model='LLAVA'):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.model = model

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        
        if self.model == 'LLAVA':
            if self.model_config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if self.model == 'LLAVA':
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        else:
            image = Image.open(os.path.join(self.image_folder, image_file))
            # image.load()
            transform = transforms.Compose([
            transforms.ToTensor()
            ])
            image_tensor = transform(image) #.squeeze(0) #.load()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, model='LLAVA'):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, model)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if args.pretrained_rob_path == 'None':
        args.pretrained_rob_path = None
    print(f"Model at: {args.pretrained_rob_path}")
    print(f"Need to load llava")
    
    if args.eval_model == 'LLAVA':
        model, image_processor, tokenizer, context_len = load_pretrained_model(model_path, args.model_base, model_name, pretrained_rob_path=args.pretrained_rob_path)
    else:
        _, image_processor, tokenizer, context_len = load_pretrained_model(model_path, args.model_base, model_name, pretrained_rob_path=args.pretrained_rob_path)
        model_args = get_of_args(args.pretrained_rob_path)
        eval_model = EvalModelAdv(model_args, adversarial=False)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        device_id = 0
        eval_model.set_device(device_id)
        # model.config = None

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor if args.eval_model == 'LLAVA' else None, model.config if args.eval_model == 'LLAVA' else None, model=args.eval_model)

    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]


        if args.eval_model == 'LLAVA':
            stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
            input_ids = input_ids.to(device='cuda', non_blocking=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=128,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            predictions = outputs.strip()
        
        else:
            transs = transforms.ToPILImage()
            ims = []
            ims.append(transs(image_tensor.squeeze()))
            image_tensor = []
            image_tensor.append(ims)
            batch_images = eval_model._prepare_images(image_tensor)
            batch_text = []
            yes_no = random.choice(['yes', 'no'])
            add_str_1 = 'Is there some object in the image?'
            add_str_2 = 'Is the image taken during day time?'
            context_text = f"Question:{add_str_1} answer:{yes_no}<|endofchunk|>"
            context_text += f"Question:{add_str_2} answer:{yes_no}<|endofchunk|>"
            context_text += f"Question:{cur_prompt} answer:"
            # Keep the text but remove the image tags for the zero-shot case
            # if num_shots == 0:
            #     context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=cur_prompt)
            )
            # print(cur_prompt)
            # batch_text.append(cur_prompt)
            outputs = eval_model.get_outputs(
                batch_images=batch_images,
                batch_text=batch_text,
                min_generation_length=0,
                max_generation_length=1,
                num_beams=3,
                length_penalty=-2.0,
            )
            dataset_name = 'coco'
            process_function = (
                postprocess_ok_vqa_generation
                if dataset_name == "ok_vqa"
                else postprocess_vqa_generation
            )

            new_predictions = map(process_function, outputs) #.strip()
            predictions = []
            for new_prediction, sample_id in zip(new_predictions, cur_prompt):
                predictions.append(new_prediction)
                # outputs = outputs.strip()
            predictions = predictions[0].strip()
            # print(predictions)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": predictions,
                                   "answer_id": ans_id,
                                   "model_id": model_name if args.eval_model == 'LLAVA' else args.eval_model,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()


    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--pretrained_rob_path", type=str, default='openai', help='Pass None, openai or path-to-rob-ckpt')
        # "/data/naman_deep_singh/project_multimodal/clip-finetune/sbatch/ViT-L-14_openai_imagenet_txtSup_False_vit-l-unsup-clean-0p1-eps4-3adv-lr1e-4-wd-1e-3_f8o0v/checkpoints/final.pt")
        # /mnt/nsingh/project_multimodal/models/ViT-L-14_openai_imagenet_txtSup_False_vit-l-unsup-clean-0p1-eps4-3adv-lr1e-4-wd-1e-3_f8o0v/checkpoints/final.pt
    parser.add_argument("--eval-model", type=str, default='LLAVA')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    eval_model(args)
