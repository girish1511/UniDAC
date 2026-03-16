"""
This script prepares the split files for the Hypersim dataset in the NYU format.
The split is at 80% train, 10% val, 10% test, at the scene level.
The hypersim dataset need to be downloaded from omnidata tools.

1. Install:
conda install -c conda-forge aria2
pip install 'omnidata-tools'

2. Follow the instruction here to download the dataset and check structure:
https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/dataset#modalities

Specifically, we use

omnitools.download point_info rgb depth_euclidean mask_valid --components hypersim --subset fullplus --dest ./omnidata_hypersim --name your-name --email your-email --agree_all

the --subset to 'fullplus' is the full dataset, ~80GB.

3. Run this script to prepare the split files in NYU format. 
If the origianl split files provided by the ominidata tools at
https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch/data/splits
Put those in 'code/data/datapath' so that this script will convert them to the NYU format.

4. Copy the split files saved in 'code/data/datapath' to the actual dataset folder for other methods to use.
"""

import os
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
    # hypersim has fixed focal length
    focal = 886.81

    print('Processing Hypersim')
    # load in hypersim provided ground-truth for comparison
    data_split_path = 'split/hypersim'
    data_path = 'datasets/hypersim'
    
    
    ################## Train subset ##################
    split_list = ['train', 'val', 'test']
    for split in split_list:
        print(f'Processing {split} subset')
        split_file = os.path.join(data_split_path, f'{split}_hypersim_orig.csv')
        save_file1 = os.path.join(data_path, f'hypersim_{split}.txt')
        
        df_valid = pd.read_csv(os.path.join(data_split_path, 'train_val_test_hypersim.csv'))
        valid_scene_name_list = df_valid.get('id').values
        df = pd.read_csv(split_file)
        scene_name_list = df.get('scene_name')
        camera_name_list = df.get('camera_name')
        frame_id_list = df.get('frame_id')

        file1 = open(save_file1, "w")
        for i in range(len(scene_name_list)):
            print(f'process {i} / {len(scene_name_list)} samples')
            # some scenes are not included in data
            if scene_name_list[i] in valid_scene_name_list:
                rgb_path = os.path.join('rgb', 'hypersim',
                                        scene_name_list[i] + '-' + camera_name_list[i],
                                        f'point_{frame_id_list[i]}_view_0_domain_rgb.png')
                # point_info_path = rgb_path.replace('domain_rgb', 'domain_point_info').replace('rgb/', 'point_info/').replace('.png', '.json')
                depth_path_euclid = rgb_path.replace('domain_rgb', 'domain_depth_euclidean').replace('rgb/',
                                                                                                'depth_euclidean/')
                # depth_path_zbuffer = rgb_path.replace('domain_rgb', 'domain_depth_zbuffer').replace('rgb/',
                #                                                                               'depth_zbuffer/')
                mask_path = rgb_path.replace('domain_rgb', 'domain_mask_valid').replace('rgb/', 'mask_valid/')

                if os.path.exists(os.path.join(data_path, rgb_path)) and os.path.exists(os.path.join(data_path, depth_path_euclid)) and os.path.exists(os.path.join(data_path, mask_path)):
                    file1.write(f'{rgb_path} {depth_path_euclid} {focal}\n')