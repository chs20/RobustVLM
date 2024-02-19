import json
import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
        which_gt=None,
        best_gt_caption_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]

        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue

            self.annotations.append(full_annotations[i])

        if isinstance(which_gt, str):
            self.which_gt = int(which_gt) if which_gt.isdigit() else which_gt
        else:
            self.which_gt = which_gt

        if best_gt_caption_path is not None:
            with open(best_gt_caption_path, 'r') as f:
                self.best_gt_captions = json.load(f)
        else:
            self.best_gt_captions = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
                if self.annotations[idx]["filepath"] == "train2014"
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["filename"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
            )
        image.load()

        image_id = self.annotations[idx]["cocoid"] if self.dataset_name == "coco" else self.annotations[idx]["filename"].split(".")[0]

        if isinstance(self.which_gt, int):
            cpt_idx = self.which_gt
        elif isinstance(self.which_gt, dict):
            cpt_idx = self.which_gt[image_id]
        elif self.which_gt == "best":
            cpt_idx = self.best_gt_captions[str(image_id)]
        else:
            assert self.which_gt is None
            cpt_idx = 0

        caption = self.annotations[idx]["sentences"][cpt_idx]["raw"]
        return {
            "image": image,
            "caption": caption,
            "image_id": image_id,
        }


class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name, which_gt='all', is_tensor=False
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}
        self.which_gt = which_gt
        self.is_tensor = is_tensor

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def get_from_id(self, question_id):
        assert not self.is_train
        assert self.dataset_name == "textvqa"
        prefix = ''
        image_path = f"{self.image_dir_path}/{prefix}{str(question_id).zfill(12)}.pt"
        image = torch.load(image_path)
        return image

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        if self.is_tensor:
            image_path = img_path.replace("jpg", "pt")
            image = torch.load(image_path)
        else:
            image = Image.open(img_path)
            image.load()
        results = {
            "image": image,
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            answers = [a["answer"] for a in answers["answers"]]
            if self.which_gt in ["all", None]:
                results["answers"] = answers
            elif isinstance(self.which_gt, int) or isinstance(self.which_gt, dict):
                which_gt = self.which_gt[question["question_id"]] if isinstance(self.which_gt, dict) else self.which_gt
                # return the nth most common answer
                counter = Counter(answers)
                most_common = counter.most_common()
                if which_gt >= len(most_common):
                    results["answers"] = []
                else:
                    results["answers"] = [most_common[which_gt][0]]
            else:
                raise ValueError(f"Unknown which_gt: {self.which_gt}")

        return results


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }


class HatefulMemesDataset(Dataset):
    def __init__(self, image_dir_path, annotations_path):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        image = Image.open(img_path)
        image.load()
        return {
            "id": annotation["id"],
            "image": image,
            "ocr": annotation["text"],
            "class_name": "yes" if annotation["label"] == 1 else "no",
            "class_id": annotation["label"],
        }


class TensorCaptionDataset(CaptionDataset):
    def get_from_id(self, image_id):
        assert self.dataset_name == "coco"
        assert not self.is_train
        # prefix = 'COCO_val2014_'
        prefix = ''
        image_path = f"{self.image_val_dir_path}/{prefix}{str(image_id).zfill(12)}.pt"
        image = torch.load(image_path)
        return image

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image_path = os.path.join(
                self.image_train_dir_path if self.annotations[idx]["filepath"] == "train2014" else self.image_val_dir_path,
                self.annotations[idx]["filename"]
            )
            image_path = image_path.replace("jpg", "pt")
            image = torch.load(image_path)
        elif self.dataset_name == "flickr":
            raise NotImplementedError
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
            )
        caption = self.annotations[idx]["sentences"][0]["raw"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
        }