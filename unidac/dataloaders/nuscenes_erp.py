import os
import json
import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset, resize_for_input
from unidac.utils.erp_geometry import cam_to_erp_patch_fast


class NuScenesERPDataset(BaseDataset):
    min_depth = 0.01
    max_depth = 80
    test_split = "nuscenes_val.json"
    train_split = "nuscenes_val.json"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        is_dense=False,
        benchmark=False,
        augmentations_db={},
        # masked=True,
        normalize=True,
        erp=True, # indicate whether the dataset is treated as erp (originally erp dataset can be treated as perspective for evaluation matching virtual f to tgt_f)
        tgt_f = 0, # focal length of perspective training data
        cano_sz=(1400, 1400), # half erp size of erp training data
        fwd_sz = (375, 1242),
        visual_debug=False,
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

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        split_path = os.path.join('splits/nuscenes', self.split_file)
        print(split_path)
        
        with open(split_path) as f:
            print(self.base_path)
            data_dict = json.load(f)
            for data in data_dict['files']:
                img_info = dict()
                img_path = data['rgb']
                depth_path = data['depth']

                img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_path)

                img_info["image_filename"] = os.path.join(self.base_path, img_path)
                img_info["camera_intrinsics"] = data['cam_in']
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
        cam_in = self.dataset[idx]["camera_intrinsics"]
        cam_intrinsics = cam_in
        # convert depth from zbuffer to euclid
        x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        depth = depth * np.sqrt((x - cam_in[2])**2 + (y - cam_in[3])**2 + cam_in[0]**2) / cam_in[0]
        depth = depth.astype(np.float32)    
        
        im_width, im_height = image.shape[1], image.shape[0]
        # print(image.shape)
        cam_params = {
            'dataset': 'nuscenes',
            'wFOV': np.arctan(im_width / 2 / cam_in[0]) * 2,
            'hFOV': np.arctan(im_height / 2 / cam_in[1]) * 2,
            'width': im_width,
            'height': im_height,
            "fx": cam_in[0],
            "fy": cam_in[1],
            "cx": cam_in[2],
            "cy": cam_in[3]
        }
            
        theta = 0
        phi = 0
        roll = 0
        crop_w = int(self.cano_sz[0] * (cam_params['wFOV'] + 0.628) / np.pi) # determine the crop size in ERP based on FOV with some padding
        # make crop aspect ratio same as fwd_sz aspect ratio. This make later visualization can directly change erp size for equivalent process
        crop_h = int(crop_w * self.fwd_sz[0] / self.fwd_sz[1])
        
        # convert image to erp patch
        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > self.min_depth
    
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            crop_h, crop_w, self.cano_sz[0], self.cano_sz[0]*2, cam_params, roll, scale_fac=None
        )
        lat_range = np.array([float(np.min(latitude)), float(np.max(latitude))])
        long_range = np.array([float(np.min(longitude)), float(np.max(longitude))])        
    
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask, long_grid, lat_grid = resize_for_input(image, depth, self.fwd_sz, cam_intrinsics, [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask, lat_grid=latitude, long_grid=longitude)
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

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, height_end = 0, self.fwd_sz[0]
        width_start, width_end = 0, self.fwd_sz[1]
        height_start, width_start = int(image.shape[0] - self.fwd_sz[0]), int((image.shape[1] - self.fwd_sz[1]) / 2)
        height_end, width_end = height_start + self.fwd_sz[0], width_start + self.fwd_sz[1]
        
        image = image[height_start:height_end, width_start:width_end]
        info["camera_intrinsics"][2] = info["camera_intrinsics"][2] - width_start
        info["camera_intrinsics"][3] = info["camera_intrinsics"][3] - height_start
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"]
            depth = depth[height_start:height_end, width_start:width_end]
            mask = depth > self.min_depth
            if self.test_mode:
                mask = np.logical_and(mask, depth < self.max_depth)
                mask = np.logical_and(mask, self.eval_mask(mask))
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
        if "long_grid" in gts:
            long_grid = gts["long_grid"]
            if long_grid is not None:
                long_grid = long_grid[height_start:height_end, width_start:width_end]
                new_gts["long_grid"] = long_grid
        return image, new_gts, info

