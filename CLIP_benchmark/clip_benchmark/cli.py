"""Console script for clip_benchmark."""
import argparse
import random
import sys
import json
from collections import defaultdict

import numpy as np
import torch
import csv
from copy import copy
import os
from torchvision import transforms
from torchvision.transforms.transforms import Compose, Resize
from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, get_dataset_default_task, dataset_collection, get_dataset_collection_from_file
from clip_benchmark.metrics import image_caption_selection, zeroshot_classification, zeroshot_retrieval, linear_probe, captioning
from clip_benchmark.model_collection import get_model_collection_from_file, model_collection
from clip_benchmark.models import load_clip, MODEL_TYPES

def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+", help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file where each line is a dataset name")
    parser_eval.add_argument('--dataset_root', default="root", type=str, help="dataset root folder where the datasets are downloaded. Can be in the form of a template depending on dataset name, e.g., --dataset_root='datasets/{dataset}'. This is useful if you evaluate on multiple datasets.")
    parser_eval.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--model', type=str, default="ViT-B-32-quickgelu", help="Model architecture to use from OpenCLIP")
    parser_eval.add_argument('--pretrained', type=str, default="laion400m_e32", help="Model checkpoint name to use from OpenCLIP")
    parser_eval.add_argument('--pretrained_model', type=str, default="", nargs="+", help="Pre-trained model(s) to use. Can be the full model name where `model` and `pretrained` are comma separated (e.g., --pretrained_model='ViT-B-32-quickgelu,laion400m_e32'), a model collection name ('openai' or 'openclip_base' or 'openclip_multilingual' or 'openclip_all'), or path of a text file where each line is a model fullname where model and pretrained are comma separated (e.g., ViT-B-32-quickgelu,laion400m_e32). --model and --pretrained are ignored if --pretrained_model is used.")
    parser_eval.add_argument('--task', type=str, default="auto", choices=["zeroshot_classification", "zeroshot_retrieval", "linear_probe", "captioning", "image_caption_selection", "auto"], help="Task to evaluate on. With --task=auto, the task is automatically inferred from the dataset.")
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=False, help="whether to use mixed precision")  # we set default to False, as we don't want amp for attacks
    parser_eval.add_argument('--num_workers', default=4, type=int)
    parser_eval.add_argument('--recall_k', default=[5], type=int, help="for retrieval, select the k for Recall@K metric. ", nargs="+",)
    parser_eval.add_argument('--fewshot_k', default=-1, type=int, help="for linear probe, how many shots. -1 = whole dataset.")
    parser_eval.add_argument('--fewshot_epochs', default=10, type=int, help="for linear probe, how many epochs.")
    parser_eval.add_argument('--fewshot_lr', default=0.1, type=float, help="for linear probe, what is the learning rate.")
    parser_eval.add_argument("--skip_load", action="store_true", help="for linear probes, when everything is cached, no need to load model.")
    parser_eval.add_argument('--seed', default=0, type=int, help="random seed.")
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument('--model_cache_dir', default=None, type=str, help="directory to where downloaded models are cached")
    parser_eval.add_argument('--feature_root', default="features", type=str, help="feature root folder where the features are stored.")
    parser_eval.add_argument('--annotation_file', default="", type=str, help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser_eval.add_argument('--custom_classname_file', default=None, type=str, help="use custom json file with classnames for each dataset, where keys are dataset names and values are list of classnames.")
    parser_eval.add_argument('--custom_template_file', default=None, type=str, help="use custom json file with prompts for each dataset, where keys are dataset names and values are list of prompts. For instance, to use CuPL prompts, use --custom_template_file='cupl_prompts.json'")

    parser_eval.add_argument('--language', default="en", type=str, nargs="+", help="language(s) of classname and prompts to use for zeroshot classification.")
    parser_eval.add_argument('--output', default="result.json", type=str, help="output file where to dump the metrics. Can be in form of a template, e.g., --output='{dataset}_{pretrained}_{model}_{language}_{task}.json'")
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false", help="suppress verbose messages")
    parser_eval.add_argument('--save_clf', default=None, type=str, help="optionally save the classification layer output by the text tower")
    parser_eval.add_argument('--load_clfs', nargs='+', default=[], type=str, help="optionally load and average mutliple layers output by text towers.")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true", help="whether to skip an evaluation if the output file exists.")
    parser_eval.add_argument('--model_type', default="open_clip", type=str, choices=MODEL_TYPES, help="clip model type")
    parser_eval.add_argument('--wds_cache_dir', default=None, type=str, help="optional cache directory for webdataset only")
    parser_eval.add_argument('--n_samples', default=-1, type=int, help="number of samples to evaluate on. -1 = whole dataset.", choices=[-1, 11, 1000])

    parser_eval.add_argument('--interpolate', default=False, action="store_true", help="interpolate with clean model")
    parser_eval.add_argument('--beta', default=0.5, type=float, help="interpolate with clean model, 0=clean")
    parser_eval.add_argument('--attack', default='none', type=str, help="attack to use", choices=['none', 'aa'])
    parser_eval.add_argument('--norm', default='Linf', type=str, help="norm to use")
    parser_eval.add_argument('--eps', default=1., type=float, help="epsilon to use")
    parser_eval.add_argument('--iterations_adv', default=100, type=int, help="number of attack iterations to use")

    parser_eval.set_defaults(which='eval')
    parser_build = subparsers.add_parser('build', help='Build CSV from evaluations')
    parser_build.add_argument('files', type=str,  nargs="+", help="path(s) of JSON result files")
    parser_build.add_argument('--output', type=str,  default="benchmark.csv", help="CSV output file")
    parser_build.set_defaults(which='build')

    args = parser.parse_args()
    return args

def main():
    base = get_parser_args()
    if base.which == "eval":
        main_eval(base)
    elif base.which == "build":
        main_build(base)

def main_build(base):
    # Build a benchmark single CSV file from a set of evaluations (JSON files)
    rows = []
    fieldnames = set()
    for path in base.files:
        data = json.load(open(path))
        row = {}
        row.update(data["metrics"])
        row.update(data)
        del row["metrics"]
        row['model_fullname'] = row['model'] + ' ' + row['pretrained']
        for field in row.keys():
            fieldnames.add(field)
        rows.append(row)
    with open(base.output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main_eval(base):
    # Get list of pre-trained models to evaluate
    pretrained_model = _as_list(base.pretrained_model)
    if pretrained_model:
        models = []
        for name in pretrained_model:
            if os.path.isfile(name):
                # if path, read file, each line is a pre-trained model
                models.extend(get_model_collection_from_file(name))
            elif name in model_collection:
                # if part of `model_collection`, retrieve from it
                models.extend(model_collection[name])
            else:
                # if not, assume it is in the form of `model,pretrained`
                model, pretrained = name.split(',')
                models.append((model, pretrained))
    else:
        models = [(base.model, base.pretrained)]
    
    # Ge list of datasets to evaluate on
    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            # If path, read file, each line is a dataset name
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            # if part of `dataset_collection`, retrieve from it
            datasets.extend(dataset_collection[name])
        else:
            # if not, assume it is simply the name of the dataset
            datasets.append(name)
    
    # Get list of languages to evaluate on
    languages = _as_list(base.language)

    if base.verbose:
        print(f"[Models] {models}")
        print(f"[Datasets] {datasets}")
        print(f"[Languages] {languages}")

    for model, pretrained in models:
        for i, dataset in enumerate(datasets):
            print(f"\n{i+1} / {len(datasets)}")
            for language in languages:
                # We iterative over all possible model/dataset/languages
                # TODO: possibility to parallelize evaluation here
                args = copy(base)
                args.model = model
                args.pretrained = pretrained
                args.dataset = dataset
                args.language = language
                run(args)

def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l

def interpolate_state_dict(m1, beta):
    m = {}

    m2 = torch.load("/path/to/ckpt.pt", map_location='cpu')
    for k in m1.keys():
        # print(m1[k].shape, m2[k].shape)
        m[k] = beta * m1[k] + (1 - beta) * m2[k]
    return m

def run(args):
    print("[args]", args, "\n")
    """Console script for clip_benchmark."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # set seed.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    task = args.task
    if args.dataset.startswith("wds/"):
        dataset_name = args.dataset.replace("wds/", "", 1)
    elif args.dataset.startswith("#"):
        print(f"Skip commented dataset {args.dataset}")
        return
    else:
        dataset_name = args.dataset
    if task == "auto":
        task = get_dataset_default_task(dataset_name)
    pretrained_slug = (
        args.pretrained.split('/')[-1] if os.path.isfile(args.pretrained) else args.pretrained
    )
    if len(pretrained_slug) > 180:
        pretrained_slug = pretrained_slug[140:]
    pretrained_slug_full_path = args.pretrained.replace('/', '_') if os.path.isfile(args.pretrained) else args.pretrained
    dataset_slug = dataset_name.replace('/', '_')
    output = args.output.format(
        model=args.model,
        attack=args.attack,
        eps=str(int(args.eps)),
        iterations=args.iterations_adv,
        pretrained=pretrained_slug,
        pretrained_full_path=pretrained_slug_full_path,
        task=task, 
        dataset=dataset_slug,
        n_samples=args.n_samples,
        language=args.language,
        bs=args.batch_size,
        beta=args.beta if args.interpolate else None,
    )
    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return
    if args.verbose:
        print(f"[Dataset] {args.dataset}")
        print(f"[Task] {task} [model] {args.pretrained} [language] {args.language}")
        print(f"[Output] {output}")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))
    if args.skip_load:
        model, transform, collate_fn, dataloader = None, None, None, None
    else:
        if args.interpolate:
            inter_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))
            inter_dict = interpolate_state_dict(inter_dict, args.beta)

        model, transform, tokenizer = load_clip(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained if not args.interpolate else inter_dict,
            cache_dir=args.model_cache_dir,
            device=args.device
        )
        if ("cifar10" in args.dataset) or ("cifar100" in args.dataset) or ("stl10" in args.dataset):
            # compute robustness wrt. original resolution
            transform_unnorm = transforms.transforms.ToTensor()
            resize = Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None)
        else:
            transform_unnorm = Compose(transform.transforms[:-1])  # remove normalize
            resize = None
        normalize = transform.transforms[-1]
        del transform  # make sure we don't use it by accident
        print(f"[Transform unnorm] {transform_unnorm}")
        print(f"[Normalize] {normalize}")

        model.eval()
        dataset = build_dataset(
            dataset_name=args.dataset, 
            root=dataset_root, 
            transform=transform_unnorm,
            split=args.split, 
            annotation_file=args.annotation_file,
            download=True,
            language=args.language,
            task=task,
            custom_template_file=args.custom_template_file,
            custom_classname_file=args.custom_classname_file,
            wds_cache_dir=args.wds_cache_dir,
        )
        if args.n_samples > 0:
            dataset = dataset.shuffle(10000, initial=10000, rng=random.Random(args.seed))
        collate_fn = get_dataset_collate_fn(args.dataset)
        if args.verbose:
            try:
                print(f"Dataset size: {len(dataset)}")
            except TypeError:
                print("IterableDataset has no len()")
            print(f"Dataset split: {args.split}")
            if hasattr(dataset, "classes") and dataset.classes:
                try:
                    print(f"Dataset classes: {dataset.classes[:20]}...")
                    print(f"Dataset number of classes: {len(dataset.classes)}")
                except AttributeError:
                    print("Dataset has no classes.")

        if args.dataset.startswith("wds/"):
            if args.n_samples > 0:
                assert args.batch_size == 50, "Otherwise we get different samples"
            dataloader = torch.utils.data.DataLoader(
                dataset.batched(args.batch_size), batch_size=None, 
                shuffle=False, num_workers=args.num_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, 
                shuffle=False, num_workers=args.num_workers, 
                collate_fn=collate_fn
            )
    if task == "zeroshot_classification":
        zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
        if args.verbose:
            print(f"Zero-shot templates: {zeroshot_templates}")
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
        if args.attack is None:
            attack_config = {
                "attack": "none",
                "bs": args.batch_size,
                "n_samples": args.n_samples,
            }
        else:
            attack_config = {
                "attack": args.attack,
                "norm": args.norm,
                "eps": args.eps,
                "iterations": args.iterations_adv,
                "bs": args.batch_size,
                "n_samples": args.n_samples,
            }
        print(f"Attack config: {attack_config}")
        metrics = zeroshot_classification.evaluate(
            model, 
            dataloader, 
            tokenizer, 
            classnames, zeroshot_templates,
            normalize=normalize,
            resize=resize,
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
            save_clf=args.save_clf,
            load_clfs=args.load_clfs,
            attack_config=attack_config,
        )
    elif task == "zeroshot_retrieval":
        metrics = zeroshot_retrieval.evaluate(
            model, 
            dataloader, 
            tokenizer, 
            recall_k_list=args.recall_k,
            device=args.device, 
            amp=args.amp
        )
    elif task == "image_caption_selection":
        metrics = image_caption_selection.evaluate(
            model,
            dataloader,
            tokenizer,
            device=args.device,
            amp=args.amp,
        )
    elif task == "linear_probe":
        # we also need the train split for linear probing.
        train_dataset = build_dataset(
            dataset_name=args.dataset, 
            root=dataset_root, 
            transform=transform, 
            split='train', 
            annotation_file=args.annotation_file,
            download=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers, 
            collate_fn=collate_fn, pin_memory=True,
        )
        metrics = linear_probe.evaluate(
            model,
            train_dataloader, 
            dataloader, 
            args.fewshot_k,
            args.batch_size,
            args.num_workers,
            args.fewshot_lr,
            args.fewshot_epochs,
            (args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
            args.seed,
            args.feature_root,
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
        )
    elif task == "captioning":
        metrics = captioning.evaluate(
            model=model, 
            dataloader=dataloader, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
            transform=transform
        )
    else:
        raise ValueError("Unsupported task: {}. task should be `zeroshot_classification`, `zeroshot_retrieval`, `linear_probe`,  or `captioning`".format(task))
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "beta": args.beta if args.interpolate else None,
        "task": task,
        "metrics": metrics,
        "language": args.language,
        "attack": args.attack,
        "iterations_adv": args.iterations_adv,
        "eps": args.eps,
        "norm": args.norm,
    }
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
