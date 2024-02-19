import sys

from train.datasets import COCOFlickrDataset, ImageNetDataset
from CLIP_eval.eval_utils import load_clip_model

sys.path.append("open_flamingo")
import os
import shutil
import time
import string
import random

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.scheduler import cosine_lr
from torchvision import transforms
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
from train.pgd_train import pgd
from train.apgd_train import apgd_train as apgd
import wandb
from train.utils import init_wandb, AverageMeter
from train.sam_data import SamData
from open_flamingo.eval.models.utils import unwrap_model
from train.utils import str2bool

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--clip_model_name', type=str, default='ViT-L-14', help='ViT-L-14, ViT-B-32')
parser.add_argument('--pretrained', type=str, default='openai')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--template', type=str, default='std')
parser.add_argument('--imagenet_root', type=str, default='/mnt/datasets/imagenet', help='Imagenet dataset root directory')
parser.add_argument('--output_normalize', type=str2bool, default=False, help='Whether the embedding is normalized')
parser.add_argument('--start_step', type=int, default=0, help='Start step for training')
parser.add_argument('--optimizer_state', type=str, default='', help='Optimizer state file path')
parser.add_argument('--steps', type=int, default=20000, help='Number of training steps')
parser.add_argument('--warmup', type=int, default=14000, help='Warmup steps')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--loss', type=str, default='l2', help='ce, l2')
parser.add_argument('--loss_clean', type=str, default='none', help='ce, l2')
parser.add_argument('--clean_weight', type=float, default=0., help='Weight for clean loss')
parser.add_argument('--trades', type=str2bool, default=False, help='Use TRADES')
parser.add_argument('--opt', type=str, default='adamw', help='Optimizer type; sgd, adamw')
parser.add_argument('--momentum_sgd', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--attack', type=str, default='apgd', help='Adversarial attack type')
parser.add_argument('--inner_loss', type=str, default='l2', help='Inner loss function for adversarial training')
parser.add_argument('--norm', type=str, default='linf', help='Norm for adversarial perturbation')
parser.add_argument('--eps', type=float, default=4, help='Epsilon for adversarial perturbation')
parser.add_argument('--iterations_adv', type=int, default=10, help='Iterations for adversarial attack')
parser.add_argument('--stepsize_adv', type=float, default=1., help='Step size for adversarial attack (no effect for apgd)')
parser.add_argument('--wandb', type=str2bool, default=True, help='Use Weights & Biases for logging')
parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--overwrite', type=str2bool, default=False, help='Overwrite existing directory')
parser.add_argument('--log_freq', type=int, default=1, help='Logging frequency')
parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency')
parser.add_argument('--output_dir', type=str, default='', help='Output directory')
parser.add_argument('--save_checkpoints', type=str2bool, default=True, help='Save 10 training checkpoints')
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')


def main(args):
    # setup wandb
    if args.wandb:
        init_wandb(
            project_name='clip-finetune',
            model_name=args.finetuned_model_name,
            config=vars(args)
        )
    else:
        wandb.init(mode='disabled')

    # print args
    print(f"Arguments:\n{'-' * 20}")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(f"{'-' * 20}")

    # setup dirs
    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=False)

    # write args to file
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    main_device = 0
    # get models
    model_orig, _, image_processor = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained='openai'
    )
    if args.optimizer_state != '':
        assert args.start_step > 0
        assert str(args.start_step) in args.optimizer_state
        assert args.pretrained in ['', 'none']
        args.pretrained = args.optimizer_state.replace('_opt', '')
    model, _, _ = load_clip_model(args.clip_model_name, args.pretrained)

    # Remove the Normalize transform by creating a new Compose object
    preprocessor_without_normalize = transforms.Compose(image_processor.transforms[:-1])
    normalize = image_processor.transforms[-1]
    del image_processor
    print(f'[preprocessor_without_normalize] {preprocessor_without_normalize}')
    print(f'[normalize] {normalize}')
    # preprocessor_without_normalize contains following transforms:
    # - Resize(size=224, interpolation=bicubic, max_size=None, antialias=warn)
    # - CenterCrop(size=(224, 224))
    # - ToTensor()
    # normalize:
    # Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    # get data
    if args.dataset == 'imagenet':
        dataset = ImageNetDataset(
            root=args.imagenet_root + '/train',
            transform=preprocessor_without_normalize,
        )

    elif args.dataset == 'segment_anything':
        dataset = SamData('/data/naman_deep_singh/datasets/newSAM', transform=preprocessor_without_normalize)

        print(dataset.__len__())
    elif args.dataset == 'coco':
        if os.path.exists('/mnt/datasets/coco'):
            image_dir_path = '/mnt/datasets/coco/train2017'
            annotations_path = '/mnt/datasets/coco/annotations/captions_train2017.json'
        elif os.path.exists('/mnt/lustre'):
            image_dir_path = '/mnt/lustre/hein/cschlarmann37/datasets/coco/train2017'
            annotations_path = '/mnt/lustre/hein/cschlarmann37/datasets/coco/annotations/captions_train2017.json'
        else:
            raise ValueError('COCO dataset not found')
        dataset = COCOFlickrDataset(
            image_dir_path=image_dir_path,
            annotations_path=annotations_path,
            transform=preprocessor_without_normalize
        )
    dataset_eval = ImageNetDataset(
        root=args.imagenet_root + '/val',
        transform=preprocessor_without_normalize,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Get text label embeddings of all ImageNet classes
    if args.template == 'std':
        template = 'This is a photo of a {}'
    elif args.template == 'blurry':
        template = 'This is a blurry photo of a {}'
    else:
        raise ValueError(f'Unknown template: {args.template}')
    print(f'template: {template}')
    texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
    text_tokens = open_clip.tokenize(texts)
    model_orig.to(main_device)
    with torch.no_grad():
        embedding_text_labels_norm = []
        for el in (text_tokens[:500], text_tokens[500:]):
            # we need to split the text tokens into two batches because otherwise we run out of memory
            # note that we are accessing the model directly here, not the CustomModel wrapper
            # thus its always normalizing the text embeddings
            embedding_text_labels_norm.append(
                model_orig.encode_text(el.to(main_device), normalize=True).detach().cpu()
            )
        embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T.to(main_device)
        assert torch.allclose(
            F.normalize(embedding_text_labels_norm, dim=0),
            embedding_text_labels_norm
        )
        if args.clip_model_name == 'ViT-B-32':
            assert embedding_text_labels_norm.shape == (512, 1000), embedding_text_labels_norm.shape
        elif args.clip_model_name in ('ViT-L-14', 'ViT-L-14-336'):
            assert embedding_text_labels_norm.shape == (768, 1000), embedding_text_labels_norm.shape
        else:
            raise ValueError(f'Unknown model: {args.clip_model_name}')

    model_orig.cpu()
    model_orig = ClipVisionModel(model=model_orig.visual, args=args, normalize=normalize)
    if num_gpus > 1:
        model_orig = torch.nn.DataParallel(model_orig)
    model_orig.cuda()

    model = ClipVisionModel(model=model.visual, args=args, normalize=normalize)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    # set optimizer (all params have requires_grad=True)
    params = unwrap_model(model).model.parameters()

    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum_sgd,
            weight_decay=args.wd
        )
    else:
        raise ValueError(f'Optimizer {args.optimizer} not supported.')
    if args.optimizer_state != '':
        optimizer.load_state_dict(torch.load(args.optimizer_state))

    # set scheduler
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)

    # compute amount of epochs
    total_epochs = args.steps / len(dataloader)
    print(f'train for {total_epochs} epochs')
    args.total_epochs = total_epochs

    # finetune
    step_total = args.start_step
    epoch = 0
    while step_total < args.steps:
        step_total = train_one_epoch(
            step_total,
            model=model,
            model_orig=model_orig,
            dataloader=dataloader,
            dataloader_eval=dataloader_eval,
            optimizer=optimizer,
            scheduler=scheduler,
            embedding_text_labels_norm=embedding_text_labels_norm,
            normalize=normalize,
            args=args,
            epoch=epoch
        )
        print(f'Epoch {epoch} done.')
        epoch += 1

    # save final model
    torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/checkpoints/final.pt')
    torch.save(optimizer.state_dict(), f'{args.output_dir}/checkpoints/final_opt.pt')

    if args.output_dir.endswith('_temp'):
        # rename temp dir to final dir
        os.rename(args.output_dir, args.output_dir[:-5])

class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, args, normalize):
        super().__init__()
        self.model = model
        self.args = args
        self.normalize = normalize

    def forward(self, vision, output_normalize):
        embedding = self.model(self.normalize(vision))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


class ComputeLossWrapper:
    def __init__(self, embedding_orig, embedding_text_labels_norm, reduction='mean', loss=None,
                 logit_scale=100.):
        self.embedding_orig = embedding_orig
        self.embedding_text_labels_norm = embedding_text_labels_norm
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale

    def __call__(self, embedding, targets):
        return compute_loss(
            loss_str=self.loss_str, embedding=embedding, targets=targets,
            embedding_orig=self.embedding_orig, logit_scale=self.logit_scale,
            embedding_text_labels_norm=self.embedding_text_labels_norm, reduction=self.reduction
            )

def train_one_epoch(
        step_total, model, model_orig, dataloader, optimizer, scheduler, normalize,
        embedding_text_labels_norm, args, epoch, dataloader_eval=None
):
    model_orig.eval()
    model.train()

    loss_meter = AverageMeter('loss')
    cos_sim_meter = AverageMeter('cos-sim')
    acc_meter = AverageMeter('acc')
    racc_meter = AverageMeter('racc')

    epoch_start_time = time.time()
    for i, (data, targets) in enumerate(dataloader):
        is_classification = isinstance(targets, torch.Tensor)
        data = data.cuda()
        n_samples = data.shape[0]
        if is_classification:
            targets = targets.cuda()

        with torch.no_grad():
            embedding_orig = model_orig(vision=data, output_normalize=args.output_normalize)

        # loss for the attack
        loss_inner_wrapper = ComputeLossWrapper(
            embedding_orig, embedding_text_labels_norm,
            reduction='none' if args.attack == 'apgd' else 'mean', loss=args.inner_loss,
            logit_scale=100.
            )
        model.eval()

        if args.attack == 'pgd':
            data_adv = pgd(
                forward=model,
                loss_fn=loss_inner_wrapper,
                data_clean=data,
                targets=targets,
                norm=args.norm,
                eps=args.eps,
                iterations=args.iterations_adv,
                stepsize=args.stepsize_adv,
                output_normalize=args.output_normalize,
                perturbation=torch.zeros_like(data).uniform_(-args.eps, args.eps).requires_grad_(True),
                mode='max',
                verbose=False
            )
        elif args.attack == 'apgd':
            # apgd currently always applies output normalization
            data_adv = apgd(
                model=model,
                loss_fn=loss_inner_wrapper,
                x=data,
                y=targets,
                norm=args.norm,
                eps=args.eps,
                n_iter=args.iterations_adv,
                verbose=True
            )
        elif args.attack == 'none':
            data_adv = data

        del loss_inner_wrapper
        model.train()

        embedding_clean = model(data, output_normalize=args.output_normalize)
        if args.clean_weight > 0.:
            loss_clean = compute_loss(
                loss_str=args.loss_clean, embedding=embedding_clean, targets=targets,
                embedding_orig=embedding_orig, logit_scale=100., embedding_text_labels_norm=None
                )
        else:
            loss_clean = 0.

        embedding_adv = model(data_adv, output_normalize=args.output_normalize)
        del data, data_adv

        if args.trades:
            embedding_clean_no_grad = embedding_clean.detach().clone()
            embedding_orig.cpu()

        loss = compute_loss(
            loss_str=args.loss, embedding=embedding_adv, targets=targets,
            embedding_orig=embedding_orig if not args.trades else embedding_clean_no_grad,
            logit_scale=100., embedding_text_labels_norm=embedding_text_labels_norm
            )
        loss_total = args.clean_weight * loss_clean + (1 - args.clean_weight) * loss
        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_total += 1
        scheduler(step_total)

        with torch.no_grad():
            # only for logging
            embedding_orig.cuda()
            cos_sim_clean = F.cosine_similarity(embedding_clean, embedding_orig, dim=1).mean()
            cos_sim = F.cosine_similarity(embedding_adv, embedding_orig, dim=1).mean()
            if is_classification:
                logits_adv = embedding_adv @ embedding_text_labels_norm
                racc = compute_acc(logits_adv, targets)
                embedding_clean_norm = F.normalize(embedding_clean, dim=1)
                logits_clean = embedding_clean_norm @ embedding_text_labels_norm
                acc = compute_acc(logits_clean, targets)
                acc_meter.update(acc, n_samples)
                racc_meter.update(racc, n_samples)
                del embedding_clean_norm, embedding_clean
            else:
                acc = None
                racc = None

        loss_meter.update(loss.item(), n_samples)
        cos_sim_meter.update(cos_sim.item(), n_samples)

        eval_logs = dict()
        if (step_total-1) % args.eval_freq == 0:
            # we compute acc and racc (against supervised apgd) on validation data
            model.eval()
            data_eval, targets_eval = next(iter(dataloader_eval))
            data_eval, targets_eval = data_eval.cuda(), targets_eval.cuda()
            loss_eval_wrapper = ComputeLossWrapper(
                embedding_orig=None, embedding_text_labels_norm=embedding_text_labels_norm,
                reduction='none', loss='ce', logit_scale=100.
                )
            data_eval_adv = apgd(
                model=model,
                loss_fn=loss_eval_wrapper,
                x=data_eval,
                y=targets_eval,
                norm=args.norm,
                eps=args.eps,
                n_iter=50,
                initial_stepsize=0.05 * args.eps if args.clean_weight > 0 else None,
                verbose=False
            )
            with torch.no_grad():
                embedding_adv_eval_norm = model(data_eval_adv, output_normalize=True)  # we set output_normalize to True
                logits_eval_adv = embedding_adv_eval_norm @ embedding_text_labels_norm
                racc_eval = compute_acc(logits_eval_adv, targets_eval)
                embedding_eval_norm = model(data_eval, output_normalize=True)
                logits_eval = embedding_eval_norm @ embedding_text_labels_norm
                acc_eval = compute_acc(logits_eval, targets_eval)
                # note we compute the cosine sim between clean and adv embedding,
                # not between orig and adv embedding as for training
                cos_sim_eval = F.cosine_similarity(embedding_adv_eval_norm, embedding_eval_norm, dim=1).mean()
            eval_logs['eval/racc'] = racc_eval
            eval_logs['eval/acc'] = acc_eval
            eval_logs['eval/cos-sim'] = cos_sim_eval
            print(f'[eval-acc] {acc_eval:.2f} [eval-racc] {racc_eval:.2f} [eval-cos-sim] {cos_sim_eval:.3f}')
            model.train()
            del data_eval_adv, data_eval, targets_eval, embedding_adv_eval_norm, logits_eval_adv, embedding_eval_norm, logits_eval

        lr_ = optimizer.param_groups[0].get('lr')
        if (step_total-1) % args.log_freq == 0:
            log_str = f'[step] {step_total} [lr] {lr_:.6f} [loss] {loss.item():.6f} [cos-sim] {cos_sim.item():.3f}'
            if is_classification:
                log_str += f' [acc] {acc:.2f} [racc] {racc:.2f}'
            print(log_str)
            log_data = {
                'step': step_total,
                'lr': lr_,
                'loss': loss.item(),
                'loss-total': loss_total.item(),
                'cos-sim-clean': cos_sim_clean.item(),
                'cos-sim': cos_sim.item(),
                'acc': acc,
                'racc': racc,
                'avg/loss': loss_meter.avg,
                'avg/cos-sim': cos_sim_meter.avg,
                'avg/acc': acc_meter.avg,
                'avg/racc': racc_meter.avg,
            }
            log_data.update(eval_logs)
            if (step_total-1) % (args.log_freq * 10) == 0:
                # compute expected average epoch time in hours
                batch_average_time = (time.time() - epoch_start_time) / (i + 1) / (60**2)
                epoch_average_time = batch_average_time * len(dataloader)
                this_epoch_remaining = epoch_average_time - \
                                       (time.time() - epoch_start_time) / 60**2
                total_remaining = epoch_average_time * (args.total_epochs - epoch - i / len(dataloader))
                print(f'[epoch average time] {epoch_average_time:.2f} [this epoch remaining] '
                      f'{this_epoch_remaining:.2f} [total remaining] {total_remaining:.2f}')

                log_data.update({
                    'time/total-remaining': total_remaining,
                    'time/this-epoch-remaining': this_epoch_remaining,
                    'time/epoch-average-time': epoch_average_time,
                    'time/batch-average-time': batch_average_time,
                    'other/epoch': epoch + i / len(dataloader),
                })
            wandb.log(log_data)

        # save 10 models over the course of training
        if args.save_checkpoints and (step_total % (args.steps // 10) == 0):
            # save model and optimizer state_dict
            torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/checkpoints/step_{step_total}.pt')
            torch.save(optimizer.state_dict(), f'{args.output_dir}/checkpoints/step_{step_total}_opt.pt')
        # every 200 steps, save a fallback model, which gets overwritten
        if step_total % 200 == 0:
            torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/checkpoints/fallback_{step_total}.pt')
            torch.save(optimizer.state_dict(), f'{args.output_dir}/checkpoints/fallback_{step_total}_opt.pt')
            # remove old fallback models
            for file in os.listdir(f'{args.output_dir}/checkpoints'):
                if file.startswith('fallback') and not str(step_total) in file:
                    os.remove(f'{args.output_dir}/checkpoints/{file}')

        if step_total >= args.steps:
            break

        torch.cuda.empty_cache()
    return step_total


@torch.no_grad()
def compute_acc(logits, targets):
    preds_clean = logits.max(dim=1)[1].detach()
    acc = (preds_clean.eq(targets).sum() / targets.shape[0]).item() * 100
    return acc


def compute_loss(loss_str, embedding, targets, embedding_orig, logit_scale,
                 embedding_text_labels_norm=None, reduction='mean'):
    if loss_str == 'l2':
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'ce':
        loss = ce(
            out=embedding @ (logit_scale * embedding_text_labels_norm),
            targets=targets,
            reduction=reduction
        )
    else:
        raise ValueError(f'loss {loss_str} not supported')
    return loss

def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch

def ce(out, targets, reduction='mean'):
    # out = logits
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    assert out.shape[0] > 1

    return F.cross_entropy(out, targets, reduction=reduction)

if __name__ == '__main__':
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Parse command-line arguments
    args = parser.parse_args()
    args.eps /= 255
    args.stepsize_adv /= 255
    # make sure there is no string in args that should be a bool
    assert not any([isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values()]), f'args contains a string that should be a bool: {args}'
    assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'

    if args.devices != '':
        # set cuda visible devices
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f'Number of GPUs available: {num_gpus}')
    else:
        print('No multiple GPUs available.')

    # set model name and output dir
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    args.finetuned_model_name = f'{args.clip_model_name}_{args.pretrained}_{args.dataset}_{args.loss}_{args.dataset}_{args.experiment_name}_{random_str}'
    args.finetuned_model_name = args.finetuned_model_name.replace('/', '_')
    args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)
    # run
    main(args)