import os
import os.path as osp
import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_to_txt_file(data, output_file):
    with open(output_file, 'w') as f:
        for item in data['files']:
            f.write(f"{item['rgb']} {item['depth']}\n")

def create_intrinsics_json(train, val, output_file):
    intrinsics = {}
    for item in train['files'] + val['files']:
        rgb_file = item['rgb']
        cam_in = item['cam_in']
        intrinsics[rgb_file] = [
            [cam_in[0], 0.0, cam_in[2]],
            [0.0, cam_in[1], cam_in[3]],
            [0.0, 0.0, 1.0]
        ]
    with open(output_file, 'w') as f:
        json.dump(intrinsics, f)

if __name__ == '__main__':
    ddad_train = read_json('splits/lyft/train_annotations.json')
    ddad_val = read_json('splits/lyft/val_annotations.json')
    output_train = 'splits/lyft/nuscenes-devkit/idisc_files/lyft_train.txt'
    output_val = 'splits/lyft/nuscenes-devkit/idisc_files/lyft_val.txt'
    output_intrinsics = 'splits/lyft/nuscenes-devkit/idisc_files/lyft_intrinsics.json'

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        train_future = executor.submit(write_to_txt_file, ddad_train, output_train)
        val_future = executor.submit(write_to_txt_file, ddad_val, output_val)
        intrinsics_future = executor.submit(create_intrinsics_json, ddad_train, ddad_val, output_intrinsics)

        for future in as_completed([train_future, val_future, intrinsics_future]):
            future.result()

    # write_to_txt_file(ddad_train, output_train)
    # write_to_txt_file(ddad_val, output_val)
    # create_intrinsics_json(ddad_train, ddad_val, output_intrinsics)

    print('Done!')