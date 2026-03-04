import os

import numpy as np
import torch
import json
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .dataset import BaseDataset
from unidac.utils.erp_geometry import cam_to_erp_patch_fast


class TaskonomyERPOnlineDataset(BaseDataset):
    min_depth = 0.01
    max_depth = 80
    test_split = "taskonomy_tiny_test.txt"
    train_split = "taskonomy_tiny_train.txt"

    def __init__(
                    
        self,
        test_mode,
        base_path,
        depth_scale=512,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        crop_size=(500, 700),
        erp_height=1400,
        theta_aug_deg=0,
        phi_aug_deg=0,
        roll_aug_deg=0,
        fov_align=True,
        visual_debug=False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        # self.crop = crop
        # the size of the output ERP patch            
        self.height = crop_size[0]
        self.width = crop_size[1]
        # the size of the full ERP image
        self.erp_height = erp_height
        self.theta_aug_deg = theta_aug_deg
        self.phi_aug_deg = phi_aug_deg
        self.roll_aug_deg = roll_aug_deg
        self.fov_align = fov_align
        self.visual_debug = visual_debug

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        print(f"Loading dataset from {self.base_path}")
        with open(os.path.join('splits/taskonomy', self.split_file)) as f:
            for line in f:
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == "None":
                        self.invalid_depth_num += 1
                        continue
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map
                    )
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                if os.path.exists(img_info["image_filename"]) and os.path.exists(img_info["annotation_filename_depth"]):
                    self.dataset.append(img_info)
                else:
                    self.invalid_depth_num += 1
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
        point_info_path = self.dataset[idx]["image_filename"].replace('domain_rgb', 'domain_point_info').replace('/rgb/', '/point_info/').replace('.png', '.json')
        with open(point_info_path, 'r') as json_file:
            point_info = json.load(json_file)
        focal_length = point_info['resolution']/2/np.tan(point_info['field_of_view_rads']/2)
        rotation = point_info['camera_rotation_final']
        ex, ey, ez = rotation
        ex -= np.pi / 2
        if ex < -np.pi:
            ex += np.pi * 2
        phi = -np.array(ex).astype(np.float32)  # gt sign opposite to erp def
        roll = -ey
        # phi = -np.array(point_info['point_pitch']).astype(np.float32)
        
        cam_params = {
            'dataset': 'taskonomy',
            'wFOV': point_info['field_of_view_rads'],
            'hFOV': point_info['field_of_view_rads'],
            'width': point_info['resolution'],
            'height': point_info['resolution'],
            'fx': focal_length,
            'fy': focal_length
        }
        
        # prepare the erp patch, and associated labels
        theta = np.deg2rad(random.uniform(-self.theta_aug_deg, self.theta_aug_deg)).astype(np.float32)
        phi += np.deg2rad(random.uniform(-self.phi_aug_deg, self.phi_aug_deg)).astype(np.float32)
        roll += np.deg2rad(random.uniform(-self.roll_aug_deg, self.roll_aug_deg)).astype(np.float32)
        image = image.astype(np.float32) / 255.0
        image_cp = image.copy()
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > self.min_depth

        fov = point_info['field_of_view_rads']
        if not self.test_mode and self.fov_align:
            scale_fac =  fov / ((self.height / self.erp_height) * np.pi) * 1.2 # 1.2 is for including more black border
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
        
        if self.visual_debug:
            print(f'fov: {np.rad2deg(fov)} deg')
            print(f'scale fac: {scale_fac}')
            print(f'pitch angle: {np.rad2deg(phi)} deg')
            print(f'roll angle: {np.rad2deg(roll)} deg')

            # visualize image, gts[gt], gts[attn_mask]
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            cmap = cm.magma_r  # Use any colormap you like (e.g., 'viridis', 'plasma', 'inferno', etc.)
            norm = mcolors.Normalize(vmin=0, vmax=15)  # Set the data range for the color bar
            plt.figure()
            plt.subplot(3, 2, 1)
            plt.imshow((image.permute(1, 2, 0) - image.min()) / (image.max() - image.min()))
            plt.title("Image")
            plt.subplot(3, 2, 2)
            # gt_vis = colorize(gts["gt"].squeeze(), vmin=self.min_depth, vmax=20)
            ax = plt.imshow(gts["gt"].squeeze(), cmap=cmap, norm=norm)
            plt.colorbar(ax, label="Meter")
            plt.title("Ground Truth")
            plt.subplot(3, 2, 3)
            plt.imshow(image_cp)
            plt.title("Image Org")
            plt.subplot(3, 2, 4)
            # pred_vis = colorize(depth.squeeze(), vmin=self.min_depth, vmax=20)
            ax = plt.imshow(depth.squeeze(), cmap=cmap, norm=norm)
            plt.colorbar(ax, label="Meter")
            plt.title("Depth Org")
            plt.subplot(3, 2, 5)
            plt.imshow(gts["mask"].squeeze().bool())
            plt.title("valid Mask")
            plt.subplot(3, 2, 6)
            plt.imshow(gts["attn_mask"].squeeze().bool())
            plt.title("Attn Mask")
            plt.subplots_adjust(wspace=0.01, top=0.99, bottom=0.01, left=0.01, right=0.99)
            save_path = 'show_dirs/debug_vis'
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'{idx}_image.png'), dpi=200)
            plt.show()

        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_grid": gts["lat_grid"], 
                    "lat_range": lat_range, "long_range": long_range, 
                    "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_grid": gts["lat_grid"], 
                    "lat_range": lat_range, "long_range": long_range}

    # def get_pointcloud_mask(self, shape):
    #     mask = np.zeros(shape)
    #     height_start, height_end = 45, self.height - 9
    #     width_start, width_end = 41, self.width - 39
    #     mask[height_start:height_end, width_start:width_end] = 1
    #     return mask

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
                attn_mask = attn_mask[height_start:height_end, width_start:width_end]
                new_gts["attn_mask"] = attn_mask
        if "lat_grid" in gts:
            lat_grid = gts["lat_grid"]
            if lat_grid is not None:
                lat_grid = lat_grid[height_start:height_end, width_start:width_end]
                new_gts["lat_grid"] = lat_grid
        return image, new_gts, info

    # def eval_mask(self, valid_mask):
    #     border_mask = np.zeros_like(valid_mask)
    #     border_mask[15:465, 20:620] = 1  # prepared center region
    #     return np.logical_and(valid_mask, border_mask)
