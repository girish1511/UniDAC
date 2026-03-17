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
from scipy import io

class iBimsERPDataset(BaseDataset):
    min_depth = 0.01
    max_depth = 25
    test_split = "ibims.json"
    train_split = "ibims.json"

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
        
        # self.crop_width = int(self.cano_sz[0] * (self.nyu_cam_params['wFOV'] + 0.314) / np.pi) # determine the crop size in ERP based on FOV with some padding
        # make crop aspect ratio same as fwd_sz aspect ratio. This make later visualization can directly change erp size for equivalent process
        # self.crop_height = int(self.crop_width * fwd_sz[0] / fwd_sz[1])

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        split_path = os.path.join('splits/ibims', self.split_file)
        print(split_path)
        # json_pitch_file = os.path.join('splits/nyu', 'nyudepthv2_test_pitch_list.json')
        # if os.path.exists(json_pitch_file):
        #     self.pitch_list = json.load(open(json_pitch_file, 'r'))
        # else:
        #     self.pitch_list = None

        with open(split_path) as f:
            print(self.base_path)
            data_dict = json.load(f)
            for data in data_dict['files']:
                img_info = dict()
                img_path = data['rgb']
                depth_path = data['depth']
                mat_path = data['mat']
                img_info["mat_path"] = os.path.join(self.base_path, mat_path)
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
        mat_data = io.loadmat(self.dataset[idx]["mat_path"])
        image = mat_data["data"]["rgb"][0][0]
        depth = mat_data["data"]["depth"][0][0]
        # image = np.asarray(
        #     Image.open(self.dataset[idx]["image_filename"])
        # )
        # depth = (
        #     np.asarray(
        #         Image.open(self.dataset[idx]["annotation_filename_depth"])
        #     ).astype(np.float32)
        #     / self.depth_scale
        # )
        
        info = self.dataset[idx].copy()
        cam_in = self.dataset[idx]["camera_intrinsics"]
        # convert depth from zbuffer to euclid
        x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        depth = depth * np.sqrt((x - cam_in[2])**2 + (y - cam_in[3])**2 + cam_in[0]**2) / cam_in[0]
        depth = depth.astype(np.float32)
        
        # center in ERP space
        theta = 0
        # if self.pitch_list is not None and self.use_pitch:
        #     phi = -np.deg2rad(self.pitch_list[idx]).astype(np.float32)
        # else:
        phi = 0
        roll = 0
        im_width, im_height = image.shape[1], image.shape[0]
        
        cam_params = {
            "dataset": "ibims",
            'wFOV': np.arctan(im_width / 2 / cam_in[0]) * 2,
            'hFOV': np.arctan(im_height / 2 / cam_in[1]) * 2,
            'width': im_width,
            'height': im_height,
            "fx": cam_in[0],
            "fy": cam_in[1],
            "cx": cam_in[2],
            "cy": cam_in[3]}

        # convert image to erp patch
        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = (depth > self.min_depth) * (depth < self.max_depth)

        crop_w = int(self.cano_sz[0] * (cam_params['wFOV'] + 0.314) / np.pi) # determine the crop size in ERP based on FOV with some padding
        # make crop aspect ratio same as fwd_sz aspect ratio. This make later visualization can directly change erp size for equivalent process
        crop_h = int(crop_w * self.fwd_sz[0] / self.fwd_sz[1])
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            crop_h, crop_w, self.cano_sz[0], self.cano_sz[0]*2, cam_params, roll, scale_fac=None
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
        
        info["camera_intrinsics"][2] = info["camera_intrinsics"][2] - width_start
        info["camera_intrinsics"][3] = info["camera_intrinsics"][3] - height_start

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
