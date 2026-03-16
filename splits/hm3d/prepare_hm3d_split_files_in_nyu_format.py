"""
This script prepares the split files for the hm3d dataset in the NYU format.
The split is at 80% train, 10% val, 10% test, at the scene level.
The hm3d dataset need to be downloaded from omnidata tools.

1. Install:
conda install -c conda-forge aria2
pip install 'omnidata-tools'

2. Follow the instruction here to download the dataset and check structure:
https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/dataset#modalities

Specifically, we use

omnitools.download point_info rgb depth_euclidean mask_valid --components hm3d --subset fullplus --dest ./omnidata_hm3d --name your-name --email your-email --agree_all

hm3d seems only have the full version, including 900 scenes, taking > 4TB. Tiny version was created using only the first 40 scenes by manully deleting the rest.

3. Run this script to prepare the split files in NYU format. 
If the origianl split files provided by the ominidata tools at
https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch/data/splits
Put those in 'code/data/datapath' so that this script will convert them to the NYU format.

4. Copy the split files saved in 'code/data/datapath' to the actual dataset folder for other methods to use.
    
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
    focal = 0  # hm3d has different focal length for each image, saved separately
    print('Processing hm3d')
    data_split_path = 'splits/hm3d'
    dataset_path = 'datasets/hm3d'
    rgb_subfolder = 'rgb/hm3d'
    depth_subfolder = 'depth_euclidean/hm3d'
    save_file_train = os.path.join(data_split_path, 'hm3d_tiny_train.txt')
    save_file_val = os.path.join(data_split_path, 'hm3d_tiny_val.txt')
    save_file_test = os.path.join(data_split_path, 'hm3d_tiny_test.txt')
    
    #############################################################################################################

    # build dict including dataset structure
    dict = {}
    dirs = glob.glob(os.path.join(dataset_path, rgb_subfolder, '*'))
    for dir in dirs:
        scene_name = os.path.basename(dir)
        dict[scene_name] = []
        for file in glob.glob(os.path.join(dir, '*.png')):
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

    file_train = open(save_file_train, "w")
    file_val = open(save_file_val, "w")
    file_test = open(save_file_test, "w")
    cnt_all = 0
    cnt_invalid = 0
    for key, value in dict.items():
        for img_file in value:
            cnt_all += 1
            print(f'process {cnt_all} samples')
            rgb_path = os.path.join(rgb_subfolder,
                                    key,
                                    img_file)
            depth_path = rgb_path.replace('rgb', 'depth_euclidean')
            mask_valid_path = rgb_path.replace('rgb', 'mask_valid')
            mask_valid = cv2.imread(os.path.join(dataset_path,mask_valid_path), cv2.IMREAD_GRAYSCALE)
            h, w = mask_valid.shape
            if np.sum(mask_valid==0) > h*w*0.1:
                cnt_invalid += 1
                print(f'{mask_valid_path} has too many invalid pixels, skip.')
                continue
            
            # filter out those images with invalid camera rotation estimation
            point_info_path = os.path.join(dataset_path, rgb_path).replace('domain_rgb', 'domain_fixatedpose').replace('/rgb/', '/point_info/').replace('.png', '.json')
            with open(point_info_path, 'r') as json_file:
                point_info = json.load(json_file)
            ex, ey, ez = point_info['camera_rotation_final']
            if abs (ey) > np.pi / 2:
                cnt_invalid += 1
                print(f'{point_info_path} shows invalid pitch angle, skip.')
                continue
            
            if key in train_keys:
                file_train.write(f'{rgb_path} {depth_path} {focal}\n')
            if key in val_keys:
                file_val.write(f'{rgb_path} {depth_path} {focal}\n')
            if key in test_keys:
                file_test.write(f'{rgb_path} {depth_path} {focal}\n')
    print(f'Done. {cnt_all} images found, {cnt_invalid} invalid samples skipped.')
