"""
The dataset will convert the fisheye image to Half ERP image testing.
This version can involve camera orientation to generate ERP pitch, with more economic crop in ERP space.

In testing
- if the model is trained from perspective images, align the erp virtual focal length to the target focal length.
- if the mode is trained from erp images, the Half ERP image will be directly uses for testing. (default)

"""

import os
import numpy as np
import cv2
import torch
from PIL import Image
import json

from .dataset import BaseDataset, resize_for_input
from unidac.utils.erp_geometry import fisheye_kb_to_erp, cam_to_erp_patch_fast
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class ScanNetPPERPDataset(BaseDataset):
    min_depth = 0.01
    max_depth = 10
    test_split = "scannetpp_tiny_test_easy.txt"
    train_split = "scannetpp_tiny_train.txt"

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
        fwd_sz = (500, 750), # input size to the network, not raw image size
        visual_debug=False,
        use_pitch=False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.erp = erp
        self.tgt_f = tgt_f
        self.cano_sz = cano_sz
        self.fwd_sz = fwd_sz
        self.visual_debug = visual_debug
        self.use_pitch = use_pitch
        print(f"use_pitch: {self.use_pitch}")
        if self.crop:
            self.crop_height, self.crop_width = fwd_sz
        else:
            self.crop_width = int(cano_sz[0])
            # make crop aspect ratio same as fwd_sz aspect ratio. This make later visualization can directly change erp size for equivalent process
            self.crop_height = int(self.crop_width * fwd_sz[0] / fwd_sz[1])
        # self.crop_width = int(cano_sz[0])
        # # make crop aspect ratio same as fwd_sz aspect ratio. This make later visualization can directly change erp size for equivalent process
        # self.crop_height = int(self.crop_width * fwd_sz[0] / fwd_sz[1])

        # load annotations
        self.scene_dict = {}
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join('splits/scannetpp', self.split_file)) as f:
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
                
                # build the dictionary as nerf-studio format
                path_items = img_name.strip().split("/")
                scene_id = path_items[1]
                if scene_id not in self.scene_dict:
                    scene_transform_file = os.path.join(self.base_path, path_items[0], scene_id, path_items[2], 'nerfstudio', "transforms.json")
                    scene_info = json.load(open(scene_transform_file))
                    scene_info_new = {k:v for k, v in scene_info.items() if k not in ["frames", "test_frames"]}
                    scene_info_new["dataset"] = "scannetpp"
                    scene_info_new["transforms"] = {}
                    for scene_frame in scene_info["frames"]:
                        scene_info_new['transforms'][scene_frame["file_path"]] = scene_frame['transform_matrix']
                    for scene_frame in scene_info["test_frames"]:
                        scene_info_new['transforms'][scene_frame["file_path"]] = scene_frame['transform_matrix']
                    fisheye_grid = np.load(os.path.join(self.base_path, path_items[0], scene_id, path_items[2], 'grid_fisheye.npy'))
                    scene_info_new['fisheye_grid'] = fisheye_grid
                    self.scene_dict[scene_id] = scene_info_new
                                                 
                # ERP output patch camera intrinsics (focal length has no actual meaning, just a simulation)
                img_info["camera_intrinsics"] = torch.tensor(
                    [
                        [1 / np.tan(np.pi/self.cano_sz[1]), 0.000000e00, self.cano_sz[1]/2],
                        [0.000000e00, 1 / np.tan(np.pi/self.cano_sz[0]), self.cano_sz[0]/2],
                        [0.000000e00, 0.000000e00, 1.000000e00],
                    ]
                )
                                
                # setup pred_scale_factor due to conversion to target focal length
                if not self.erp:
                    img_info["pred_scale_factor"] = (img_info["camera_intrinsics"][0, 0] + img_info["camera_intrinsics"][1, 1]).item() / 2 / self.tgt_f
                    img_info["camera_intrinsics"][0, 0] /= img_info["pred_scale_factor"]
                    img_info["camera_intrinsics"][1, 1] /= img_info["pred_scale_factor"]
                else:
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
                cv2.imread(self.dataset[idx]["annotation_filename_depth"], cv2.IMREAD_ANYDEPTH)
            ).astype(np.float32)
            / self.depth_scale
        )
        info = self.dataset[idx].copy()
        path_items = self.dataset[idx]["image_filename"].strip().split("/")
        scene_id = path_items[3]
        
        # convert depth from zbuffer to euclid (critical for fisheye dataset to use euclid depth)
        # Because the depth is rendered as z-buffer, converting back to euclid with undistorted ray direction via the ray lookup table for efficiency
        fisheye_grid = self.scene_dict[scene_id]['fisheye_grid']
        fisheye_grid_z = cv2.resize(fisheye_grid[:, :, 2], (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
        depth = depth / fisheye_grid_z
        depth = depth.astype(np.float32)
        
        # convert fisheye image to erp image patch
        file_name = path_items[-1]
        transform_mat = np.asarray(self.scene_dict[scene_id]["transforms"][file_name])        
        # using pitch can lead to slight improvement, but sometimes intorduce artifact in corners
        if self.use_pitch:
            phi = (np.arcsin(-transform_mat[2, 2])).astype(np.float32)
            # phi = np.clip(phi, -np.pi/4, np.pi/4).astype(np.float32)  # temporal handling for pitch angle
        else:
            phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0
        
        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > self.min_depth

        cam_params = self.scene_dict[scene_id]
        # frame_num = self.dataset[idx]["annotation_filename_depth"].split('/')[-1][:-4]
        # pred_cam_params_path = os.path.join("/user/ganesang/cvlshare/cvl-ganesang/AnyCalib/pred/scannetpp", f"{frame_num}.json")
        # with open(pred_cam_params_path, 'r') as f:
        #     cam_params = json.load(f)
        #     cam_params.update({'camera_model': 'OPENCV_FISHEYE'})

        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            self.crop_height, self.crop_width, self.cano_sz[0], self.cano_sz[0]*2, cam_params, roll, scale_fac=None
        )
        lat_range = np.array([float(np.min(latitude)), float(np.max(latitude))])
        long_range = np.array([float(np.min(longitude)), float(np.max(longitude))])
        info["camera_intrinsics"] = self.dataset[idx]["camera_intrinsics"].clone()
        
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask, lat_grid, long_grid = resize_for_input((image * 255.).astype(np.uint8), depth, self.fwd_sz, info["camera_intrinsics"], [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask, lat_grid=latitude, long_grid=longitude)

        info['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor
        info['pad'] = pad
        info['scene_id'] = scene_id
        info['theta'] = theta
        info['phi'] = phi
        info['roll'] = roll
        if not self.test_mode:
            depth /= info['pred_scale_factor']
    
        image, gts, info = self.transform(image=image, gts={"depth": depth, "attn_mask": (attn_mask>0).astype(np.float32), "lat_grid": lat_grid, "long_grid": long_grid}, info=info)
        
        if self.visual_debug:
            print(transform_mat)
            print(f'pitch angle: {np.rad2deg(phi)} deg')
            print(f'roll angle: {np.rad2deg(roll)} deg')
            # visualize image, gts[gt], gts[attn_mask]
            import matplotlib.pyplot as plt
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
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_range": lat_range, "long_range":long_range, "lat_grid": gts["lat_grid"], "long_grid": gts["long_grid"], "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_range": lat_range, "long_range":long_range, "lat_grid": gts["lat_grid"]}
        
    # def get_pointcloud_mask(self, shape):
    #     mask = np.zeros(shape)
    #     height_start, height_end = 45, self.height - 9
    #     width_start, width_end = 41, self.width - 39
    #     mask[height_start:height_end, width_start:width_end] = 1
    #     return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, height_end = 0, self.fwd_sz[0]
        width_start, width_end = 0, self.fwd_sz[1]
        image = image[height_start:height_end, width_start:width_end]
        # info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        # info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start

        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"][height_start:height_end, width_start:width_end]
            mask = depth > self.min_depth
            if self.test_mode:
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
        if "long_grid" in gts:
            long_grid = gts["long_grid"]
            if long_grid is not None:
                long_grid = long_grid[height_start:height_end, width_start:width_end]
                new_gts["long_grid"] = long_grid
        return image, new_gts, info

    # def eval_mask(self, valid_mask):
    #     border_mask = np.zeros_like(valid_mask)
    #     border_mask[15:465, 20:620] = 1  # prepared center region
    #     return np.logical_and(valid_mask, border_mask)
