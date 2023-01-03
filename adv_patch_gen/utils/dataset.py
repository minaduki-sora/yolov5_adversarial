"""
Dataset Class for loading YOLO format datasets where the source data dir has the image and labels subdirs
where each image must have a corresponding label file with the same name
"""
import glob
import os.path as osp
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTNS = {".png", ".jpg", ".jpeg"}


class YOLODataset(Dataset):
    """Create a dataset for adversarial-yolt.

    Attributes:
        image_dir: Directory containing the images of the YOLO format dataset.
        label_dir: Directory containing the labels of the YOLO format dataset.
        max_labels: max number labels to use for each image
        imgsize: (width, height) of image
        shuffle: Whether or not to shuffle the dataset.
    """

    def __init__(self, image_dir: str, label_dir: str, max_labels: int, imgsize: Tuple[int, int], shuffle: bool = True):
        image_paths = glob.glob(osp.join(image_dir, "*"))
        label_paths = glob.glob(osp.join(label_dir, "*"))
        image_paths = sorted(
            [p for p in image_paths if osp.splitext(p)[-1] in IMG_EXTNS])
        label_paths = sorted(
            [p for p in label_paths if osp.splitext(p)[-1] in {".txt"}])

        assert len(image_paths) == len(
            label_paths), "Number of images and number of labels don't match"
        # all corresponding image and labels must exist
        for img, lab in zip(image_paths, label_paths):
            if osp.basename(img).split('.')[0] != osp.basename(lab).split('.')[0]:
                raise FileNotFoundError(
                    f"Matching image {img} or label {lab} not found")
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.imgsize = imgsize
        self.shuffle = shuffle
        self.max_n_labels = max_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx <= len(self), "Index range error"
        img_path = self.image_paths[idx]
        lab_path = self.label_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # check to see if label file contains any annotation data
        if osp.getsize(lab_path):
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        image = transforms.ToTensor()(image)
        label = self.pad_label(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """
        Pad image and adjust label
        """
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        resize = transforms.Resize(self.imgsize)
        padded_img = resize(padded_img)  # choose here

        return padded_img, lab

    def pad_label(self, label: torch.Tensor) -> torch.Tensor:
        """
        Pad labels if fewer labels than max_n_labels present
        """
        pad_size = self.max_n_labels - label.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(label, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = label[:self.max_n_labels]
        return padded_lab

