import sys
import open_clip
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import transforms

from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL



def print_statistics(arr):
    # make sure its 1-d
    assert len(arr.shape) == 1
    print(f'[mean] {arr.mean():.4f} [median] {np.median(arr):.4f} [min] {arr.min():.4f} [max] '
          f'{arr.max():.4f} [std] {arr.std():.4f} [n] {len(arr)}\n')

def interpolate_state_dict(m1, beta=0.):
    m = {}
    try:
        m2 = torch.load("/mnt/nsingh/project_multimodal/models/clip-vit-l-visual.pt", map_location='cpu')
    except:  
        m2 = torch.load("/data/naman_deep_singh/project_multimodal/clip-vit-l-visual.pt", map_location='cpu')
    for k in m1.keys():
        m[k] = (1 - beta) * m1[k] +  beta * m2[k]

    return m


def load_clip_model(clip_model_name, pretrained, beta=0.):
    try:  # try loading only visual model
        model, _, image_processor = open_clip.create_model_and_transforms(
            clip_model_name, pretrained='openai', device='cpu'
        )
        if pretrained != 'openai':
            if isinstance(pretrained, str):
                checkpoint = torch.load(pretrained, map_location=torch.device('cpu'))
            else:
                checkpoint = pretrained
            # if beta non-zero interpolate between clean and pretrained ckpts
            if beta != 0.:
                print("beta", beta)
                checkpoint = interpolate_state_dict(pretrained, beta)

            if 'vision_encoder_state_dict' in checkpoint.keys():  # tecoa checkpoint
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
            else:
                model.visual.load_state_dict(checkpoint)
    except RuntimeError as e:  # try loading whole model
        print(f'error: {e}', file=sys.stderr)
        print('retrying by loading whole model..', file=sys.stderr)
        torch.cuda.empty_cache()
        model, _, image_processor = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained, device='cpu'
        )
    model.eval()

    # Remove the Normalize transform by creating a new Compose object
    preprocessor_no_norm = transforms.Compose(image_processor.transforms[:-1])
    normalizer = image_processor.transforms[-1]
    return model, preprocessor_no_norm, normalizer

@torch.no_grad()
def get_text_embeddings(model, dataset, texts):
    assert not (dataset and texts)
    if dataset:
        assert dataset == 'imagenet'
    if dataset == 'imagenet':
        template = 'This is a photo of a {}'
        texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
        text_tokens = open_clip.tokenize(texts)
    elif texts:
        text_tokens = open_clip.tokenize(texts)
    embedding_text_labels_norm = []
    chunk_size = 500
    for i in range(0, len(text_tokens), chunk_size):
        el = text_tokens[i:i+chunk_size]
        embedding_text_labels_norm.append(
            model.model.encode_text(el.cuda(), normalize=True).detach().cpu()
        )
    embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T
    if dataset == 'imagenet':
        assert (embedding_text_labels_norm.shape == (512, 1000)
                or embedding_text_labels_norm.shape == (768, 1000)), embedding_text_labels_norm.shape
    return embedding_text_labels_norm


@torch.inference_mode()
def compute_accuracy_no_dataloader(model, data, targets, device, batch_size=1000):
    # data, targets: tensors
    # (in parts copied from autoattack)
    train_flag = model.training
    model.eval()
    n_batches = int(np.ceil(data.shape[0] / batch_size))
    n_total = 0
    n_correct = 0
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, data.shape[0])
        data_batch = data[start_idx:end_idx, :].clone().to(device)
        targets_batch = targets[start_idx:end_idx].clone().to(device)
        logits = model(data_batch)
        confs, preds = F.softmax(logits, dim=1).max(dim=1)
        n_total += targets_batch.size(0)
        n_correct += (preds.eq(targets_batch).sum()).item()
    acc = n_correct / n_total

    # print(f'{n_total=}')
    # print(f'{n_correct=}')
    if train_flag:
        model.train()
    return acc


