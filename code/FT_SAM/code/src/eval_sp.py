# -------------------------------------------------------------------------------
# Name:        eval.py
# Purpose:     Evaluate different models segmentation performance
#
# Author:      Fangyijie Wang
#
# Created:     19/06/2024
# Copyright:   (c) Fangyijie Wang (2024)
# Licence:     MIT
# -------------------------------------------------------------------------------

import time
import argparse
import torch
from transformers import SamModel, SamProcessor
from utils import Datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))


def show_boxes_on_image(raw_image, boxes, mask, gt, dsc):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.7])
    color2 = np.array([255 / 255, 165 / 255, 0 / 255, 0.3])
    for box in boxes:
      show_box(box, plt.gca())
    ax = plt.gca()
    h, w = mask.shape[-2:]

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    gt_image = gt.reshape(h, w, 1) * color2.reshape(1, 1, -1)
    ax.imshow(gt_image)
    ax.imshow(mask_image)

    #ax.title.set_text(f"DSC:{dsc:.3f}")
    plt.xlabel('', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.title(f"DSC:{dsc:.3f}", fontsize=16)
    plt.axis('on')
    plt.show()


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def main():

    current_time = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir", type=str, default='', help="Root directory of dataset.")

    parser.add_argument(
        "--img_dir", type=str, default='', help="Directory of testing images.")

    parser.add_argument(
        "--logs_dir", type=str, default='', help="Directory of testing logs.")

    parser.add_argument(
        "--model", type=str, default="sam", help="Model to evaluate {sam, medsam, ftsam, sythsam.")
    
    parser.add_argument(
        '--save_fig', type=bool, default=False, help='Enable saving predictions')    

    parser.add_argument(
        '--debug', action='store_true', help='Enable debug mode')

    parser.add_argument(
        "--eval_iter", type=int, default=5, help="Number of times to evaluate model.")

    parser.add_argument(
        "--samsize", type=str, default="base", help="Size of SAM model to evaluate {base, large, huge.")

    parser.add_argument(
        "--checkpoint", type=str, help="Use a model checkpoint for evaluation.")
    
    parser.add_argument(
        "--image_size", type=int, default=256, help="Resolution of image data.")    

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if args.model == 'medsam':
        processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
        model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
    elif args.model == 'sam':
        if args.samsize == 'base':
            model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
            processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        elif args.samsize == 'large':
            model = SamModel.from_pretrained("facebook/sam-vit-large").to(device)
            processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
        elif args.samsize == 'huge':
            model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
            processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    elif args.model == 'ftsam':
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        model_state = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(model_state)
        model.to(device)
    else:
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    ROOT_DIR = args.root_dir
    IMG_DIR = args.img_dir
    IMG_SIZE = args.image_size  # 256
    # create predictions dir
    preds_fpath = f"{ROOT_DIR}/{IMG_DIR}/pred/"

    if not args.debug:
        if os.path.exists(preds_fpath):
            # shutil.rmtree(preds_fpath)
            print(f"Directory {preds_fpath} already existing, deleting.")

        os.makedirs(os.path.dirname(preds_fpath), exist_ok=True)

    save_fig = args.save_fig
    dsc_scores = []
    logs = {}
    for iter_i in range(args.eval_iter):
        print(f"Evaluating model for iteration {iter_i}...")

        # create dataset. Note there is a random perturbation component to the bounding box generation so results
        # will be slightly different for each dataset evaluation run
        # Spanish
        dataset = Datasets.SpAfDataset(image_dir=f"{ROOT_DIR}", 
                                       mask_dir=f"{ROOT_DIR}", 
                                       split="test",
                                       region="ES",
                                       img_size=IMG_SIZE,
                                       nobox=False)

        dsc_vals = []
        i = 1
        for sample in dataset:
            if i % 100 == 0:
              print(f"Processing sample {i}/{len(dataset)}")
            i += 1

            filename = f'{args.model}_{sample["filename"]}'
            image = sample['image']
            input_boxes = sample['prompt']

            # prepare image + box prompt for the model
            inputs = processor(image, input_boxes=[[input_boxes]], return_tensors="pt").to(torch.float32).to(device)

            # forward pass
            # note that the authors use `multimask_output=False` when performing inference
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)

            # apply sigmoid
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


            # post processs the masks to reshape them to match the original image size
            #masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                                 #inputs["reshaped_input_sizes"].cpu())


            #if args.model == "medsam":
                # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb
                # Note that MedSAM was fine-tuned using a custom DiceWithSigmoid loss, so we need to apply the appropriate postprocessing here:
                # apply sigmoid
               # medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                # convert soft mask to hard mask
                #medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
                #medsam_seg_h = (medsam_seg_prob > 0.5).astype(np.uint8)

                #tseg = torch.tensor(medsam_seg_h)
                #tseg = tseg.unsqueeze(0)
                #tseg = tseg.unsqueeze(0)
                #tseg = tseg.unsqueeze(0)

                #masks = processor.image_processor.post_process_masks(tseg,
                                                                     #inputs["original_sizes"].cpu(),
                                                                     #inputs["reshaped_input_sizes"].cpu())
            # we only create one output mask, so just need first item in list
            #medsam_seg = masks[0]

            # reshape the mask to match ground truth
            #h, w = medsam_seg.shape[-2:]
            #mask_image = medsam_seg.reshape(h, w)

            ground_truth_seg = (np.array(sample['mask']) / 255).astype(int)
            dsc = dice_coefficient(ground_truth_seg, medsam_seg)

            # if save_fig:
            #     # save plot of prediction over-layed on image
            #     fig, axes = plt.subplots()
            #     axes.imshow(np.array(image))
            #     show_mask(medsam_seg, axes)
            #     axes.title.set_text(f"Predicted mask - DSC:{dsc:.3f}")
            #     axes.axis("off")

            #     if not args.debug:
            #         plt.savefig(f'{ROOT_DIR}/{IMG_DIR}/pred/{filename}')
            #         plt.cla()

            dsc_vals.append(dsc)
            #print(f'DSC:{dsc} ---- {filename}')
            #show_boxes_on_image(image, [input_boxes], medsam_seg, ground_truth_seg, dsc)
        
        # plt.close(fig)

        # calculate mean DSC
        #print(f'DSC_vals: {dsc_vals}')
        avg_dsc = np.mean(dsc_vals)
        dsc_scores.append(avg_dsc)
        print(f'Average DSC: {avg_dsc}')

        # Some log information to help you keep track of your model information.
        logs = {
            "model": args.model,
            "region": 'Spanish',
            "checkpoint": args.checkpoint,
            "dsc_scores": dsc_scores,
            "mean_dsc": float(np.mean(dsc_scores)),
            "std_dsc": float(np.std(dsc_scores)),
        }

    SUB_LOGS = args.logs_dir
    if args.model == "sam":
        with open(os.path.join(f'./logs/{SUB_LOGS}', f"{args.model}_SP_{args.samsize}_{current_time}_{args.eval_iter}.json"), 'w') as f:
            json.dump(logs, f)
    else:
        with open(os.path.join(f'./logs/{SUB_LOGS}', f"{args.model}_SP_{current_time}_{args.eval_iter}.json"), 'w') as f:
            json.dump(logs, f)

if __name__ == '__main__':
    main()
