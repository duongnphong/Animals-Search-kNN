import os
import numpy as np
from tqdm import tqdm

def read(featuredir):
    list_npy = []
    list_label = []
    for class_names in os.listdir(featuredir):
        dirpath = os.path.join(featuredir, class_names)
        for i in os.listdir(dirpath):
            featurepath = os.path.join(dirpath, i)
            list_npy.append(featurepath)
            list_label.append(class_names)
    return list_npy, list_label

def load_feature(featuredir):
    list_npy, list_label = read(featuredir)

    list_feature = []
    for i in tqdm(list_npy):
        list_feature.append(np.load(i).squeeze(0))
    
    return list_feature, list_label, list_npy

if __name__ == "__main__":
    load_feature()