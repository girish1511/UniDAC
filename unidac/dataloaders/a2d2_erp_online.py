import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset
from unidac.utils.erp_geometry import cam_to_erp_patch_fast
import torch


class A2D2ERPOnlineDataset(BaseDataset):
    min_depth = 0.01
    max_depth = 100
    test_split = "a2d2_val.json"
    train_split = "a2d2_train.json"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        is_dense=False,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        crop_size=(500, 700),
        erp_height=1400,
        theta_aug_deg=0,
        phi_aug_deg=10,
        roll_aug_deg=0,
        # rescale=1.5,
        fov_align=True,
        visual_debug=False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.is_dense = is_dense
        self.height = crop_size[0] #500 #450 #900 #256
        self.width = crop_size[1] #650 #600 #1180 
        # self.height_start, self.width_start = 180, 8
        # self.height_end, self.width_end = (
        #     self.height_start + self.height,
        #     self.width_start + self.width,
        # )
        self.erp_height = erp_height
        self.theta_aug_deg = theta_aug_deg
        self.phi_aug_deg = phi_aug_deg
        self.roll_aug_deg = roll_aug_deg
        self.visual_debug = visual_debug
        self.fov_align = fov_align
        self.fov_avg=[0,0]
        self.fov_min = np.inf
        self.fov_max = -np.inf
        # self.rescale = rescale
        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        print(f"Loading dataset from {self.base_path}")
        with open(os.path.join('splits/a2d2', self.split_file)) as f:
            data_dict = json.load(f)
            for data in data_dict['files']:
                img_info = dict()
                img_path = data['rgb']
                depth_path = data['depth']
                img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_path
                    )
                img_info["image_filename"] = os.path.join(self.base_path, img_path)
                img_info["cam_intrinsics"] = data['cam_in'] 
                self.dataset.append(img_info)
            # for line in f:
            #     img_info = dict()
            #     img_name = line.strip().split(" ")[0]

            #     if not self.benchmark:  # benchmark test
            #         depth_map = line.strip().split(" ")[1]
            #         img_info["annotation_filename_depth"] = os.path.join(
            #             self.base_path, depth_map
            #         )
            #     img_info["image_filename"] = os.path.join(self.base_path, img_name)
            #     self.dataset.append(img_info)
        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )
    
    def __getitem__(self, idx):
        image = np.asarray(Image.open(self.dataset[idx]["image_filename"]))[...,:3]
        # if not self.benchmark:
        depth = (
            np.asarray(Image.open(self.dataset[idx]["annotation_filename_depth"])).astype(
                np.float32
            )
            / self.depth_scale
        )
        cam_intrinsics = self.dataset[idx]["cam_intrinsics"]
        cam_params = {
            "dataset": "a2d2",
            "wFOV": np.arctan(1920 / 2 / cam_intrinsics[0]) * 2,
            "hFOV": np.arctan(1208 / 2 / cam_intrinsics[1]) * 2,
            "width": 1920,
            "height": 1208, 
            "fx": cam_intrinsics[0],
            "fy": cam_intrinsics[1],
        }

        if depth is not None:
            x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
            depth = depth * np.sqrt((x - cam_intrinsics[2])**2 + (y - cam_intrinsics[3])**2 + cam_intrinsics[0]**2) / cam_intrinsics[0]
            depth = depth.astype(np.float32)

        theta = np.deg2rad(np.random.uniform(-self.theta_aug_deg, self.theta_aug_deg)).astype(np.float32)
        phi = np.deg2rad(np.random.uniform(-self.phi_aug_deg, self.phi_aug_deg)).astype(np.float32)

        roll = np.deg2rad(np.random.uniform(-self.roll_aug_deg, self.roll_aug_deg)).astype(np.float32)
        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = (depth > self.min_depth) & (depth < self.max_depth)

        if not self.test_mode and self.fov_align:
            # scale_fac = cam_params["hFOV"] / ((self.height / self.erp_height)*np.pi)
            scale_fac =  cam_params["hFOV"] / ((self.height / self.erp_height) * np.pi) * 1.2 # 1.2 is for including more black border
        else:
            scale_fac = 1.0

        erp_rgb, erp_depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
        image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
        self.height, self.width, self.erp_height, self.erp_height*2, cam_params, roll, scale_fac=scale_fac
        )

        lat_range = np.array([float(np.min(latitude)), float(np.max(latitude))])
        long_range = np.array([float(np.min(longitude)), float(np.max(longitude))])

        info = self.dataset[idx].copy()
        # ERP output patch camera intrinsics (focal length has no actual meaning, just a simulation)
        info["camera_intrinsics"] = torch.tensor(
            [
                [1 / np.tan(np.pi/self.erp_height), 0.000000e00, self.width/2],
                [0.000000e00, 1 / np.tan(np.pi/self.erp_height), self.height/2],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )

        # Image augmentation. Should only include those compatible with ERP
        image, gts, info = self.transform(image=(erp_rgb * 255.).astype(np.uint8), gts={"depth": erp_depth, "attn_mask": erp_mask, "lat_grid": latitude}, info=info)


        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_grid": gts["lat_grid"],
                    "lat_range": lat_range, "long_range": long_range, 
                    # "intrinsics": info["camera_intrinsics"], "scale": info.get("scale", 1.0), "phi": phi, "theta": theta,
                    "info": info,
                    }
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_grid": gts["lat_grid"],
                    "lat_range": lat_range, "long_range": long_range, 
                    # "intrinsics": info["camera_intrinsics"], "scale": info.get("scale", 1.0), "phi": phi, "theta": theta
                    }

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, width_start = int(image.shape[0] - self.height), int(
            (image.shape[1] - self.width) / 2
        )
        height_end, width_end = height_start + self.height, width_start + self.width
        image = image[height_start:height_end, width_start:width_end]
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"]
            if depth is not None:
                height_start, width_start = int(depth.shape[0] - self.height), int(
                    (depth.shape[1] - self.width) / 2
                )
                height_end, width_end = (
                    height_start + self.height,
                    width_start + self.width,
                )
                depth = depth[height_start:height_end, width_start:width_end]
                mask = depth > self.min_depth
                # if self.test_mode:
                mask = np.logical_and(mask, depth < self.max_depth)
                    # mask = self.eval_mask(mask)
                mask = mask.astype(np.uint8)
                new_gts["gt"] = depth
                new_gts["mask"] = mask
        if "attn_mask" in gts:
            attn_mask = gts["attn_mask"]
            if attn_mask is not None:
                height_start, width_start = int(attn_mask.shape[0] - self.height), int(
                    (attn_mask.shape[1] - self.width) / 2
                )
                height_end, width_end = (
                    height_start + self.height,
                    width_start + self.width,
                )
                attn_mask = attn_mask[height_start:height_end, width_start:width_end]
                new_gts["attn_mask"] = attn_mask
        if "lat_grid" in gts:
            lat_grid = gts["lat_grid"]
            if lat_grid is not None:
                lat_grid = lat_grid[height_start:height_end, width_start:width_end]
                new_gts["lat_grid"] = lat_grid

        return image, new_gts, info