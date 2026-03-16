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
    # tgt sizes calulated by the ratio of focal length (886.81 / 519)
    import yaml
    focal = 162.97  # approximated at the equatorial
    print('Processing Gibson V2')
    data_split_path = '../code/data/datapath'

    #############################################################################################################

    # load in GV2 provided ground-truth
    full_file = '../code/data/datapath/GV2_full_v1.yaml'

    with open(full_file) as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)

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

    save_file_train = os.path.join(data_split_path, 'gv2_train.txt')
    save_file_val = os.path.join(data_split_path, 'gv2_val.txt')
    save_file_test = os.path.join(data_split_path, 'gv2_test.txt')
    file_train = open(save_file_train, "w")
    file_val = open(save_file_val, "w")
    file_test = open(save_file_test, "w")
    for key, value in dict.items():
        for img_file in value:
            rgb_path = os.path.join(
                                    key,
                                    img_file)
            depth_path = rgb_path.replace('emission', 'depth').replace('.png', '.exr')
            if key in train_keys:
                file_train.write(f'{rgb_path} {depth_path} {focal}\n')
            if key in val_keys:
                file_val.write(f'{rgb_path} {depth_path} {focal}\n')
            if key in test_keys:
                file_test.write(f'{rgb_path} {depth_path} {focal}\n')
    print('Done.')
