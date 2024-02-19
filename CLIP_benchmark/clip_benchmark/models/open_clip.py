import sys
import open_clip
import torch


def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    try:
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained='openai', cache_dir=cache_dir, device='cpu'
        )
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
        else:
            checkpoint = pretrained
        if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
            model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
        else:
            model.visual.load_state_dict(checkpoint)
    except Exception as e:
        # try loading whole model
        print(f'error: {e}', file=sys.stderr)
        print('retrying by loading whole model..', file=sys.stderr)
        torch.cuda.empty_cache()
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir, device='cpu'
        )

    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer
