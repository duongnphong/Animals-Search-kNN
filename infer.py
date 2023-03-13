# Task 2: Pre-processing images and extract features from the images by 
# letting it go through the model and convert them to .npy file 
# -> save all of them 

import torch
from torchvision.transforms import *
from model import model_eval
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# Transform image with resizing, to tensor, normalization
def transform(img):
    transform = Compose([
        Resize((224,224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

# Normalize images with L2 Normalization
def L2Normalize(feature):
    norm2 = torch.sqrt(torch.sum(torch.square(feature),dim=1))   
    feature = feature / norm2
    return feature

# Take an input of directory and return a list of all paths to all images 
def append_list(datadir):
    list_img = []
    for class_names in os.listdir(datadir):
        dirpath = os.path.join(datadir, class_names)
        dir_name = f"feature/{class_names}"
        os.makedirs(dir_name, exist_ok=True)
        for i in os.listdir(dirpath):
            impath = os.path.join(dirpath, i)
            list_img.append(impath)
    return list_img


def main():
    img_list = append_list("data/")
    model = model_eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in tqdm(img_list):
        img = Image.open(i).convert("RGB")
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        feature = model(img)
        feature = feature.cpu().detach().numpy()
        extension = i.split(".")[-1]
        saved_name = i.replace("data", "feature").replace(extension, "npy")
        np.save(saved_name, feature)
    

    # feature = torch.nn.functional.normalize(feature, p=2, dim=1)

if __name__ == "__main__":
    main()
