from torch.utils.data import Dataset
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

import torch.utils.data as data
import numpy as np
# torch.manual_seed(0)
import random
# random.seed(0)
# np.random.seed(0)

# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted([entry.name for entry in os.scandir(directory) if entry.is_dir()])
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: int(cls_name) for cls_name in (classes)}
    return classes, class_to_idx


class SamData(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = sorted(list(pathlib.Path(targ_dir).glob("*/*.jpg"))) # note: you'd have to update this if you've got .png's or .jpeg's
        # print(os.path.basename(self.paths)) 
        # Setup transforms
        self.indexes = []
        self.folds = []
        for i, n in enumerate(self.paths):
#             if i<=50:
            strrr= str(n)
#             print(strrr[strrr.index('sa_')+3:strrr.index('sa_')+9])
            self.indexes.append(int(strrr[strrr.index('sa_')+13:strrr.index('.jpg')]))
            self.folds.append(strrr[strrr.index('sa_')+3:strrr.index('sa_')+9])

        self.transform = transform
        # Create classes and class_to_idx attributes
#         self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data, label (X, y, index)."
        img = self.load_image(index)

        indx = self.indexes[index]
        # fold_i = self.folds[index]
        # print(fold_i)
#         class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
#         class_idx = self.class_to_idx[class_name]
        # Transform if necessary
        if self.transform:
            return self.transform(img), indx # return X, index)
        else:
            return img, indx # class_idx, indx # return data, label (X, y, index)