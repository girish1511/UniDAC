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

# TODO: may need to prepare exact NYU reso, but keep the content range within

if __name__ == '__main__':
    # tgt sizes calulated by the ratio of focal length (886.81 / 519)
    import yaml
    focal = 162.97  # approximated at the equatorial
    print('Processing Matterport 3D')
    data_path = 'datasets/matterport3d'

    #############################################################################################################

    # load in hypersim provided ground-truth for comparison
    split_file = '../code/data/datapath/M3D_v1_train.yaml'
    save_file = os.path.join(data_path, 'm3d_train.txt')

    with open(split_file) as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)

    file1 = open(save_file, "w")
    for key, value in dict.items():
        for img_file in value:
            rgb_path = os.path.join(
                                    key,
                                    img_file)
            depth_path = rgb_path.replace('emission', 'depth').replace('.png', '.exr')

            file1.write(f'{rgb_path} {depth_path} {focal}\n')

    #############################################################################################################

    # load in hypersim provided ground-truth for comparison
    split_file = '../code/data/datapath/M3D_v1_val.yaml'
    save_file = os.path.join(data_path, 'm3d_val.txt')

    with open(split_file) as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)

    file1 = open(save_file, "w")
    for key, value in dict.items():
        for img_file in value:
            rgb_path = os.path.join(
                                    key,
                                    img_file)
            depth_path = rgb_path.replace('emission', 'depth').replace('.png', '.exr')

            file1.write(f'{rgb_path} {depth_path} {focal}\n')

    #############################################################################################################

    # load in hypersim provided ground-truth for comparison
    split_file = '../code/data/datapath/M3D_v1_test.yaml'
    save_file = os.path.join(data_path, 'm3d_test.txt')

    with open(split_file) as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)

    file1 = open(save_file, "w")
    for key, value in dict.items():
        for img_file in value:
            rgb_path = os.path.join(
                                    key,
                                    img_file)
            depth_path = rgb_path.replace('emission', 'depth').replace('.png', '.exr')

            file1.write(f'{rgb_path} {depth_path} {focal}\n')
