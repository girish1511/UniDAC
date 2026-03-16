"""
This script prepares the split files for the hm3d dataset in the NYU format.
The split is at 80% train, 10% val, 10% test, at the scene level.
1. The scannet++ dataset need to be downloaded first.

2. Run this script to prepare the split files in NYU format. 

3. Copy the split files saved in 'code/data/datapath' to the actual dataset folder for other methods to use.
    
"""
    
import os
import random

import numpy as np
import glob
import torch
import cv2
# import open3d as o3d
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import json
import shutil

if __name__ == '__main__':
    focal = 0  # scannetpp has fisheye focal length for each image, not saved here
    print('Processing scannetpp')
    data_split_path = 'code/data/datapath'
    dataset_path = 'datasets/scannetpp'
    rgb_subfolder = 'dslr/resized_images'
    depth_subfolder = 'dslr/render_depth'
    # treat all the images from the tiny dataset as test set
    save_file_train = os.path.join(data_split_path, 'scannetpp_tiny_train.txt')
    save_file_val = os.path.join(data_split_path, 'scannetpp_tiny_val.txt')
    save_file_test = os.path.join(data_split_path, 'scannetpp_tiny_test.txt')
   
    #############################################################################################################

    # build dict including dataset structure
    dict = {}
    dirs = glob.glob(os.path.join(dataset_path, 'data/*'))
    for dir in dirs:
        scene_name = 'data/'+ os.path.basename(dir)
        dict[scene_name] = []
        for file in sorted(glob.glob(os.path.join(dir, rgb_subfolder, '*.JPG'))):
            dict[scene_name].append(os.path.basename(file))

    # split the dictionary into train, val, test
    all_keys = list(dict.keys())
    random.seed(555)
    random.shuffle(all_keys)
    num_keys = len(all_keys)
    num_train = int(num_keys/10*8)
    num_val = int(num_keys/10)
    num_test = int(num_keys/10)
    train_keys = all_keys[:num_train]
    val_keys = all_keys[num_train:num_train+num_val]
    test_keys = all_keys[num_train+num_val:]
    # test_keys = all_keys[num_train:]

    file_train = open(save_file_train, "w")
    file_val = open(save_file_val, "w")
    file_test = open(save_file_test, "w")
    for key, value in dict.items():
        for img_file in value:
            rgb_path = os.path.join(
                key,
                rgb_subfolder,                
                img_file)
            depth_path = rgb_path.replace(rgb_subfolder, depth_subfolder).replace('.JPG', '.png')
            if key in train_keys:
                file_train.write(f'{rgb_path} {depth_path} {focal}\n')
            if key in val_keys:
                file_val.write(f'{rgb_path} {depth_path} {focal}\n')
            if key in test_keys:
                file_test.write(f'{rgb_path} {depth_path} {focal}\n')
    print('Done.')
