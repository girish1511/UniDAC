"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os

import numpy as np
import torch
from PIL import Image
import json

from .dataset import BaseDataset, resize_for_input
from unidac.utils.erp_geometry import cam_to_erp_patch_fast


class NYUERPDataset(BaseDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor(
            [
                [5.1885790117450188e02, 0, 3.2558244941119034e02],
                [0, 5.1946961112127485e02, 2.5373616633400465e02],
                [0, 0, 1],
            ]
        )
    }
    nyu_cam_params = {
    'dataset': 'nyu',
    'fx': 5.1885790117450188e+02,
    'fy': 5.1946961112127485e+02,
    'cx': 3.2558244941119034e+02,
    'cy': 2.5373616633400465e+02,
    'k1': 2.0796615318809061e-01,
    'k2': -5.8613825163911781e-01,
    'p1': 7.2231363135888329e-04,
    'p2': 1.0479627195765181e-03,
    'k3': 4.9856986684705107e-01,
    'wFOV': 1.105,
    'hFOV': 0.8663
    }
    min_depth = 0.01
    max_depth = 10
    test_split = "nyu_test.txt"
    train_split = "nyu_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=1000,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        erp=True, # indicate whether the dataset is treated as erp (originally erp dataset can be treated as perspective for evaluation matching virtual f to tgt_f)
        tgt_f = 519, # focal length of perspective training data
        cano_sz=(1400, 1400), # half erp size of erp training data
        fwd_sz = (480, 640), # input size to the network, not raw image size
        visual_debug=False,
        use_pitch=False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        # self.tgt_f = tgt_f
        self.cano_sz = cano_sz
        self.fwd_sz = fwd_sz
        self.visual_debug = visual_debug
        self.use_pitch = use_pitch
        
        self.crop_width = int(self.cano_sz[0] * (self.nyu_cam_params['wFOV'] + 0.314) / np.pi) # determine the crop size in ERP based on FOV with some padding
        # make crop aspect ratio same as fwd_sz aspect ratio. This make later visualization can directly change erp size for equivalent process
        self.crop_height = int(self.crop_width * fwd_sz[0] / fwd_sz[1])

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        print(os.path.join('splits/nyu', self.split_file))
        json_pitch_file = os.path.join('splits/nyu', 'nyudepthv2_test_pitch_list.json')
        if os.path.exists(json_pitch_file):
            self.pitch_list = json.load(open(json_pitch_file, 'r'))
        else:
            self.pitch_list = None
        with open(os.path.join('splits/nyu', self.split_file)) as f:
            print(self.base_path)
            for line in f:
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == "None":
                        self.invalid_depth_num += 1
                        continue
                    if depth_map[0] == '/':
                        depth_map = depth_map[1:]
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map)
                img_name = line.strip().split(" ")[0]
                if img_name[0] == '/':
                    img_name = img_name[1:]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                img_info["camera_intrinsics"] = self.CAM_INTRINSIC["ALL"].clone()
                img_info["pred_scale_factor"] = 1.0
                self.dataset.append(img_info)

        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        image = np.asarray(
            Image.open(self.dataset[idx]["image_filename"])
        )
        depth = (
            np.asarray(
                Image.open(self.dataset[idx]["annotation_filename_depth"])
            ).astype(np.float32)
            / self.depth_scale
        )
        
        info = self.dataset[idx].copy()
        
        # convert depth from zbuffer to euclid
        x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        depth = depth * np.sqrt((x - info["camera_intrinsics"][0, 2].item())**2 + (y - info["camera_intrinsics"][1, 2].item())**2 + info["camera_intrinsics"][0, 0].item()**2) / info["camera_intrinsics"][0, 0].item()
        depth = depth.astype(np.float32)
        
        # center in ERP space
        theta = 0
        if self.pitch_list is not None and self.use_pitch:
            phi = -np.deg2rad(self.pitch_list[idx]).astype(np.float32)
        else:
            phi = 0
        roll = 0
        
        # convert image to erp patch
        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > self.min_depth
        
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            self.crop_height, self.crop_width, self.cano_sz[0], self.cano_sz[0]*2, self.nyu_cam_params, roll, scale_fac=None
        )
        lat_range = np.array([float(np.min(latitude)), float(np.max(latitude))])
        long_range = np.array([float(np.min(longitude)), float(np.max(longitude))])
        
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask,_,_ = resize_for_input(image, depth, self.fwd_sz, info["camera_intrinsics"], [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask)
        info['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor
        info['pad'] = pad
        info['phi'] = phi
        if not self.test_mode:
            depth /= info['pred_scale_factor']
            
        image, gts, info = self.transform(image=(image * 255.).astype(np.uint8), gts={"depth": depth, "attn_mask": (attn_mask>0).astype(np.float32)}, info=info)

        if self.visual_debug:
            # visualize image, gts[gt], gts[attn_mask]
            import matplotlib.pyplot as plt
            print(f'phi: {np.rad2deg(phi)} deg, theta: {np.rad2deg(theta)} deg, roll: {np.rad2deg(roll)} deg')
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow((image.permute(1, 2, 0) - image.min()) / (image.max() - image.min()))
            plt.title("Image")
            plt.subplot(2, 2, 2)
            plt.imshow(gts["gt"].squeeze())
            plt.title("Ground Truth")
            plt.subplot(2, 2, 3)
            plt.imshow(gts["mask"].squeeze().bool())
            plt.title("valid Mask")
            plt.subplot(2, 2, 4)
            plt.imshow(gts["attn_mask"].squeeze().bool())
            plt.title("Attn Mask")
            plt.show()
            
        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_range": lat_range, "long_range":long_range, "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_range": lat_range, "long_range":long_range}

    # def get_pointcloud_mask(self, shape):
    #     mask = np.zeros(shape)
    #     height_start, height_end = 45, self.fwd_sz[0] - 9
    #     width_start, width_end = 41, self.fwd_sz[1] - 39
    #     mask[height_start:height_end, width_start:width_end] = 1
    #     return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, height_end = 0, self.fwd_sz[0]
        width_start, width_end = 0, self.fwd_sz[1]
        image = image[height_start:height_end, width_start:width_end]
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start

        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"][height_start:height_end, width_start:width_end]
            mask = depth > self.min_depth
            if self.test_mode:
                mask = np.logical_and(mask, depth < self.max_depth)
                mask = self.eval_mask(mask)
            mask = mask.astype(np.uint8)
            new_gts["gt"] = depth
            new_gts["mask"] = mask
            
        if "attn_mask" in gts:
            attn_mask = gts["attn_mask"]
            if attn_mask is not None:
                attn_mask = attn_mask[height_start:height_end, width_start:width_end]
                new_gts["attn_mask"] = attn_mask
        return image, new_gts, info

    def eval_mask(self, valid_mask):
        """Do grag_crop or eigen_crop for testing"""
        border_mask = np.zeros_like(valid_mask)
        border_mask[45:471, 41:601] = 1
        return np.logical_and(valid_mask, border_mask)
