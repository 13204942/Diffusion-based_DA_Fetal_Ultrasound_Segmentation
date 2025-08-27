# -------------------------------------------------------------------------------
# Name:        ft_sam.py
# Purpose:     Fine-tune SAM using custom dataset
#
# Author:      Kevin Whelan, Fangyijie Wang
#
# Created:     20/06/2024
# Copyright:   (c) Kevin Whelan (2024)
# Licence:     MIT
# Based on: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb
# -------------------------------------------------------------------------------

from transformers import SamProcessor
from transformers import SamModel
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim import Adam
import monai
import albumentations as A
from tqdm import tqdm
import argparse
from statistics import mean
import torch
import os
from utils import Datasets
import matplotlib.pyplot as plt
import time
import json
import numpy as np
from utils.metrics import dice_score
#from pyinstrument import Profiler


def view_mask(mask):
    img = mask.detach().numpy()


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, default='', help="Root directory of dataset.")
    
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Total training epochs.")
    
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Total batch size.")
        
    parser.add_argument(
        "--train_size", type=int, default=5, help="Number of real images for fine-tuning SAM.")

    parser.add_argument(
        "--syn_size", type=int, default=0, help="Number of synthetic images for fine-tuning SAM.")
    
    parser.add_argument(
        "--image_size", type=int, default=256, help="Resolution of image data.")

    parser.add_argument(
        "--data_agumentation", type=str, default='', help="Data augmentation type: SA or WA.")
        
    args = parser.parse_args()
    # profiler = Profiler()
    # profiler.start()

    current_time = time.strftime("%Y%m%d-%H%M%S")
    LR = 1e-5
    NUM_EPOCHS = args.num_epochs  # 200
    BATCH_SIZE = args.batch_size  # 5
    ROOT_DIR = args.root_dir
    TRAIN_SIZE = args.train_size  # 300
    SYN_SIZE = args.syn_size  # 200
    IMG_SIZE = args.image_size  # 256
    # DiceMetricCalc = monai.metrics.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    SA_WA = args.data_agumentation

    start_time = time.strftime("%d%m%y_%H%M%S")
    st = time.time()
    # Weak Augmentation: Horizontal/Vertical Flipping, Rotation, Brightness and Contrast, Blur, Gaussian Noise
    wa_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),contrast_limit=(-0.5, 0.5)),  # Image-only transforms
            A.Blur(blur_limit=(3, 3), p=0.3),  # Image-only transforms
            A.GaussNoise(std_range=(0.05, 0.1), p=0.3),  # Image-only transforms
         ],
    )

    # Strong Augmentation: Strong Scaling, Color Jitter, Elastic Deformation, Random Erasing.
    sa_transforms = A.Compose(
        [
            # A.RandomScale(scale_limit=0.5, p=0.5),  # Scale between 50% and 150%
            A.Rotate(limit=45, p=0.5),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5),  # Image-only transforms
            A.ElasticTransform(alpha=120, sigma=12, p=0.5),
            A.CoarseDropout(
                        num_holes_range=(5, 15),        # 5-15 holes per object
                        hole_height_range=(0.1, 0.1),  # 10% of sqrt(object area)
                        hole_width_range=(0.1, 0.1),   # 10% of sqrt(object area)
                        fill=0,                        # Fill holes with black
                        p=0.5
                        )  # mask_fill_value is NULL, then Image-only transforms. Otherwise, Image-Mask transforms
         ],
    )

    if SA_WA == 'SA':
        dataset = Datasets.SpAfDataset(image_dir=f"{ROOT_DIR}",
                                    mask_dir=f"{ROOT_DIR}",
                                    nobox=False,
                                    num=TRAIN_SIZE,
                                    split="train",
                                    img_size=IMG_SIZE,
                                    transform=sa_transforms)
    elif SA_WA == 'WA':
        dataset = Datasets.SpAfDataset(image_dir=f"{ROOT_DIR}",
                                    mask_dir=f"{ROOT_DIR}",
                                    nobox=False,
                                    num=TRAIN_SIZE,
                                    split="train",
                                    img_size=IMG_SIZE,
                                    transform=wa_transforms)
    else:
        dataset = Datasets.SpAfDataset(image_dir=f"{ROOT_DIR}",
                                    mask_dir=f"{ROOT_DIR}",
                                    nobox=False,
                                    split="train",
                                    num=TRAIN_SIZE,
                                    img_size=IMG_SIZE)
    
    val_dataset = Datasets.SpAfDataset(image_dir=f"{ROOT_DIR}",
                                   mask_dir=f"{ROOT_DIR}",
                                   nobox=False,
                                   split="val",
                                   img_size=IMG_SIZE)

    # example = dataset[0]
    # image = example["image"]

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base",
                                             size={"longest_edge": 1024},
                                             mask_size={"longest_edge": IMG_SIZE})
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    train_dataset = Datasets.SAMDataset(dataset=dataset, processor=processor)
    val_dataset = Datasets.SAMDataset(dataset=val_dataset, processor=processor)

    # create Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # batch = next(iter(train_dataloader))
    # for k, v in batch.items():
    #    print(k, v.shape)

    # print(batch["ground_truth_mask"].shape)

    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=LR, weight_decay=0)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    num_epochs = NUM_EPOCHS

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    t_losses = []
    v_losses = []
    v_dscs = []
    best_loss = 1.0

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []
        train_dsc = []
        val_dsc = []
        model.train()
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            # predicted_masks = torch.nn.functional.interpolate(predicted_masks, size=(270, 400))
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            train_losses.append(loss.item())

        t_loss = np.mean(train_losses)
        t_losses.append(t_loss)

        model.eval()
        for batch in tqdm(val_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            # predicted_masks = torch.nn.functional.interpolate(predicted_masks, size=(270, 400))
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # print(predicted_masks.shape)
            # print(ground_truth_masks.shape)

            # dsc = DiceMetricCalc(outputs.pred_masks, ground_truth_masks)
            dsc = dice_score(predicted_masks, ground_truth_masks)

            val_losses.append(loss.item())

            val_dsc.append(torch.mean(dsc).item())
            # print(f"val_dsc: {torch.mean(dsc).item()}")

        v_loss = np.mean(val_losses)
        v_losses.append(v_loss)

        v_dsc = np.mean(val_dsc)
        v_dscs.append(v_dsc)

        print(f'EPOCH: {epoch}')
        print(f'\n\ttrain Loss: {t_loss}\ttrain dsc: {1} \n\tval Loss: {v_loss}\tval acc: {v_dsc}')

        # print(f'EPOCH: {epoch}')
        # print(f'Mean loss: {mean(epoch_losses)}')
        # losses.append(mean(epoch_losses))
        # profiler.stop()
        # profiler.print()

        # save model
        saved_name = f"/mnt/storage/fangyijie/ft_sam/checkpoints_head/ESTT/{SA_WA}/checkpoint_FTSAM_epoch_{epoch}_BS_{BATCH_SIZE}_TRAINSIZE_{TRAIN_SIZE}_Real_{current_time}"
        print(f"v_loss: {v_loss}; best_loss: {best_loss}")
        if v_loss < best_loss:
            best_loss = v_loss
            if epoch > 5:
                torch.save(model.state_dict(), saved_name)

    logs = {
        "model": "ftsam",
        "checkpoint": saved_name,
        "train_losses": t_losses,
        "val_losses": v_losses,
        "val_DSC": v_dscs,
        "best_val_epoch": int(np.argmin(v_losses) + 1),
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "train_size": TRAIN_SIZE
    }
    with open(os.path.join("./tr_logs/head", f"ftsam_{SA_WA}_{start_time}.json"), 'w') as f:
        json.dump(logs, f)

    print(f'Model is saved at: {saved_name}')
    print(f'Execution Time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st))}')


if '__main__' == __name__:
    main()
