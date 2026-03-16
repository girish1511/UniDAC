import os

import numpy as np
import cv2
import torch
from PIL import Image

from .dataset import BaseDataset, resize_for_input

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class GibsonV2Dataset(BaseDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor(
            [
                [1 / np.tan(np.pi/512), 0.000000e00, 256.],
                [0.000000e00, 1 / np.tan(np.pi/512), 512.],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
    }
    min_depth = 0.01
    max_depth = 10
    test_split = "gv2_test.txt"
    train_split = "gv2_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=1,
        crop=None,
        benchmark=False,
        augmentations_db={},
        # masked=False,
        normalize=True,
        erp=False, # indicate whether the testing model is treated as erp (originally erp dataset can be treated as perspective in evaluation by matching virtual f to tgt_f)
        tgt_f = 519, # focal length of perspective training data
        fwd_sz=(512, 1024),
        cano_sz=(1800, 3600),
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.erp = erp
        self.fwd_sz = fwd_sz

        if not self.erp:
            self.tgt_f = tgt_f
        else:
            self.cano_sz = cano_sz

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join('splits/gibson_v2', self.split_file)) as f:
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
                img_info["camera_intrinsics"] = self.CAM_INTRINSIC['ALL'][:, :3].clone()
                
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
        info["camera_intrinsics"] = self.dataset[idx]["camera_intrinsics"].clone()
        H, W = image.shape[:2]
        # latitude, longitude = torch.meshgrid(torch.linspace(-torch.pi/2, torch.pi/2, H), torch.linspace(-torch.pi, torch.pi, W))
        latitude, longitude = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, H), np.linspace(-np.pi, np.pi, W), indexing='ij')
        # resizing process to fwd_sz.
        if not self.erp:
            image, depth, pad, pred_scale_factor = resize_for_input(image, depth, self.fwd_sz, info["camera_intrinsics"], [image.shape[0], image.shape[1]], 1.0, lat_grid=latitude, long_grid=longitude)
        else:
            to_cano_ratio = self.cano_sz[0] / image.shape[0]
            image, depth, pad, pred_scale_factor, lat_grid, long_grid = resize_for_input(image, depth, self.fwd_sz, info["camera_intrinsics"], self.cano_sz, to_cano_ratio, lat_grid=latitude, long_grid=longitude)
        info['f_align_factor'] = np.copy(info['pred_scale_factor'])
        info['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor
        info['pad'] = pad
        if not self.test_mode:
            depth /= info['pred_scale_factor']
    
        image, gts, info = self.transform(image=image, gts={"depth": depth, "lat_grid": lat_grid, "long_grid": long_grid}, info=info)
        lat_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
        long_range = torch.tensor([-np.pi, np.pi], dtype=torch.float32)
        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "lat_range": lat_range, "long_range":long_range, "lat_grid": gts["lat_grid"], "long_grid": gts["long_grid"], "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "lat_range": lat_range, "long_range":long_range, "lat_grid": gts["lat_grid"], "long_grid": gts["long_grid"]}
        

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