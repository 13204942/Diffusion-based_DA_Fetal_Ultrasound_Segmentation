# -------------------------------------------------------------------------------
# Name:        Datasets.py
# Purpose:     Custom DataSet class for fetal ultrasound images
#
# Author:      Kevin Whelan, Fangyijie Wang
#
# Created:     15/06/2024
# Copyright:   (c) Kevin Whelan (2024)
# Licence:     MIT
# -------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import os
import math
import numpy as np
import pandas as pd


def resize_mask(image):
    longest_edge = 256

    # get new size
    w, h = image.size
    scale = longest_edge * 1.0 / max(h, w)
    new_h, new_w = h * scale, w * scale
    new_h = int(new_h + 0.5)
    new_w = int(new_w + 0.5)

    resized_image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return resized_image


def pad_mask(image):
    pad_height = 256 - image.height
    pad_width = 256 - image.width

    padding = ((0, pad_height), (0, pad_width))
    padded_image = np.pad(image, padding, mode="constant")
    return padded_image


def process_mask(image):
    resized_mask = resize_mask(image)
    padded_mask = pad_mask(resized_mask)
    return padded_mask


def get_bounding_box(ground_truth_map, nobox, image):
    # get bounding box from mask

    if nobox:
        # bbox = [0, 0, image.size[0], image.size[1]]
        bbox = [np.random.randint(0, 20),
                np.random.randint(0, 20),
                image.size[0] - np.random.randint(0, 20),
                image.size[1] - np.random.randint(0, 20)]
    else:
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

    return bbox


def fill_mask(mask):
    # need to check each pixel to see if it has a line to its left and its right
    mask_array = np.array(mask)
    H, W = mask_array.shape

    for i in range(H):
        for j in range(W):
            # take row slice to left and right of current pixel
            left_slice = mask_array[i, 0:j]
            right_slice = mask_array[i, j + 1:]
            if np.sum(left_slice) > 0 and np.sum(right_slice) > 0:
                mask_array[i, j] = 255

    return Image.fromarray(mask_array)


class SpAfDataset(Dataset):
    """
        Custom Dataset for Spanish and African Ultrasound Image dataset
        Consists of fetal head ultrasound image and mask pairs
    """

    def __init__(self, image_dir, mask_dir, split="train", region="ES", nobox=False, img_size=256, num=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sample_list = []
        self.region = region
        self.split = split
        self.images = os.listdir(image_dir)
        self.result = {}
        self.nobox = nobox
        self.img_size = img_size
        self.num=num
        self.transform = transform

        if self.num and self.split == "train":
            with open(self.image_dir + f"/sp_head_train/train/data_{self.num}.list", "r") as f1:
                self.sample_list = f1.readlines()

        if self.split == "val":
            with open(self.image_dir + "/sp_head_train/val/data.list", "r") as f:
                self.sample_list = f.readlines()

        if self.split == "test":
            if self.region == "AF":
                with open(self.image_dir + "/af_head/data.list", "r") as f:
                    self.sample_list = f.readlines()
            else:
                with open(self.image_dir + "/sp_head_test/data.list", "r") as f:
                    self.sample_list = f.readlines()

        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train" or self.split == "val":
            image_path = self.image_dir + "/sp_head_train/image/{}.png".format(case)
            mask_path = self.image_dir + "/sp_head_train/mask/{}_Annotation.png".format(case)
        elif self.split == "test" and self.region == "ES":
            image_path = self.image_dir + "/sp_head_test/image/{}.png".format(case)
            mask_path = self.image_dir + "/sp_head_test/mask/{}_Annotation.png".format(case)
        elif self.split == "test" and self.region == "AF":
            image_path = self.image_dir + "/af_head/image/{}.png".format(case)
            mask_path = self.image_dir + "/af_head/mask/{}_Annotation.png".format(case)
        image = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
        mask = Image.open(mask_path).convert('L').resize((self.img_size, self.img_size))

        ground_truth_seg = (np.array(mask) / 255).astype(int)
        input_boxes = get_bounding_box(ground_truth_seg, self.nobox, image)

        image = np.array(image)
        mask = np.array(mask)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        self.result["image"] = image
        self.result["mask"] = mask
        self.result["filename"] = f'{case}_pred_mask.png'
        self.result["prompt"] = input_boxes

        return self.result
    

class HC18Dataset(Dataset):
    """
        Custom Dataset for HC18 challenge Ultrasound Image dataset
        Consists of fetal head ultrasound image and mask pairs
    """

    def __init__(self, image_dir, mask_dir, split="train", nobox=False, num=None, img_size=256, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sample_list = []
        self.split = split
        self.images = os.listdir(image_dir)
        self.result = {}
        self.nobox = nobox
        self.num=num
        self.img_size = img_size
        self.transform = transform

        if self.num and self.split == "train":
            with open(self.image_dir + f"/fewshot_sam/train/data_{self.num}.list", "r") as f1:
                self.sample_list = f1.readlines()

        if self.split == "val":
            with open(self.image_dir + "/fewshot_sam/val/data.list", "r") as f:
                self.sample_list = f.readlines()
        elif self.split == "test":
            with open(self.image_dir + "/fewshot_sam/test/data.list", "r") as f:
                self.sample_list = f.readlines()

        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = self.image_dir + "/image/{}.png".format(case)
        mask_path = self.image_dir + "/mask/{}_Annotation.png".format(case)
        image = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
        mask = Image.open(mask_path).convert('L').resize((self.img_size, self.img_size))

        ground_truth_seg = (np.array(mask) / 255).astype(int)
        input_boxes = get_bounding_box(ground_truth_seg, self.nobox, image)
        image = np.array(image)
        mask = np.array(mask)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        self.result["image"] = image
        self.result["mask"] = mask
        self.result["filename"] = f'{case}_pred_mask.png'
        self.result["prompt"] = input_boxes

        return self.result


class USDataset(Dataset):
    """
    Custom Dataset for Spanish Ultrasound Image dataset
    """

    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.transform = transform
        self.class_map = {v: k for k, v in self.dataset.class_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target_id = self.dataset[index]
        target_label = self.class_map[target_id]
        # return multiple fields for the image, so we can identify filename of misclassified images
        sample = {
            "data": image,
            "target_id": target_id,
            "target_label": target_label,
            "index": index,
            "folder": os.path.dirname(self.dataset.imgs[index][0]),
            "filename": os.path.basename(self.dataset.imgs[index][0]),
        }
        return sample

    @property
    def classes(self):
        return self.dataset.classes

    # calculate the weights for weighted sampler
    def sampler_weights(self):
        target = torch.tensor(self.dataset.targets)
        class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in target])

        return samples_weight


class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = (np.array(item["mask"]) / 255).astype(int)

        # get bounding box prompt
        prompt = item["prompt"]

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt").to(torch.float32)

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs
