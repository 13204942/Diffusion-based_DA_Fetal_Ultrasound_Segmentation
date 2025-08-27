import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os


# root_dir = '/content/drive/MyDrive/colab/diffusers/my_data/hc18/lora_trim/head'
# prompt_txt = root_dir + '/prompts.txt'

def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


class USDataset(Dataset):
    def __init__(self, root_dir, image_dir, mask_dir, transform1=None, transform2=None, tokenizer=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.tokenizer = tokenizer
        self.result = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.images[idx][-3:] == 'png':
            image_path = os.path.join(self.image_dir, self.images[idx])
            # based on the name of the image get the path of the mask
            # file_prefix = self.images[idx].split('_')[0]
            # file_prefix2 = self.images[idx].split('_')[1].split('.')[0]
            # mask_filename = f'{file_prefix}_{file_prefix2}_Annotation.png'
            mask_filename = self.images[idx].replace(".png", "_Annotation.png")

            mask_path = os.path.join(self.mask_dir, mask_filename)
            image = Image.open(image_path)
            mask = Image.open(mask_path)

            if self.transform1:
                image = self.transform1(image)
                mask = self.transform1(mask)
            # combine image and mask x 2
            combined_m_i = torch.cat([image, image, mask], dim=0)

            combined_m_i = self.transform2(combined_m_i)

            self.result["pixel_values"] = combined_m_i
            # if file_prefix in ["1","2","5", "6", "9", "14", "15", "18", "27", "35", "36", "37", "38", "45", "46", "51", "64", "68", "89", "110"]:
            #     caption = "First trimester ultrasound image of a fetal head."
            # elif file_prefix in ["314", "372", "415", "458", "482", "505", "534", "609", "637", "654", "678", "683", "713", "716", "720", "732", "742", "750", "770", "772"]:
            #     caption = "Second trimester ultrasound image of a fetal head."
            # else:
            #     caption = "Third trimester ultrasound image of a fetal head."

            file = open(self.root_dir + '/prompt.txt', "r")
            caption = file.read()
            # print(caption)
            file.close()

            # caption = "An ultrasound image of a fetal head."
            # print(f"image:{self.images[idx]},caption:{caption}")
            self.result["input_ids"] = tokenize_captions(caption, self.tokenizer)

        return self.result


# # Define transformations
# # transform to apply to both mask and image before concatenating
# transform1 = transforms.Compose([
#     transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
#     # Resize images to a consistent size
#     transforms.ToTensor(),  # Convert images to tensors
# ])
#
# # transform to apply to concatenated image + mask
# transform2 = transforms.Compose([
#     transforms.Normalize([0.5], [0.5]),
# ])
#
# root_dir = '../../my_data/us/brainmask/sample'
# # Create custom dataset and data loader
# dataset = USDataset(image_dir=f'{root_dir}/image',
#                     mask_dir=f'{root_dir}/mask',
#                     transform1=transform1, transform2=transform2)
#
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
#
# for batch in data_loader:
#     masked_image = batch
#     print(masked_image.shape)