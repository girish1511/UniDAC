import os
import os.path as osp
import json
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import erp_patch_to_cam_fast, cam_to_erp_patch_fast
from unidac.utils.visualization import save_val_imgs_v2
from unidac.dataloaders.dataset import resize_for_input

OUTDOOR_SAMPLE_1 = {
    "config_file": "configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json",
    "dataset_name": "kitti360",
    "image_filename": "demo/input/kitti360_rgb.png",
    "annotation_filename_depth": "demo/input/kitti360_depth.png",
    "depth_scale": 256.0,
    "fishey_grid": "splits/kitti360/grid_fisheye_02.npy",
    "crop_wFoV": 180, # degree decided by origianl data fov + some buffer
    "fwd_sz": (704, 704), # the patch size input to the model
    "erp": False,
    "cam_params": {
        'dataset':'kitti360',
        "fx": 1.3363220825849971e+03,
        "fy": 1.3357883350012958e+03,
        "cx": 7.1694323510126321e+02,
        "cy": 7.0576498308221585e+02,
        "xi": 2.2134047507854890e+00,
        "k1": 1.6798235660113681e-02,
        "k2": 1.6548773243373522e+00,
        "p1": 4.2223943394772046e-04,
        "p2": 4.2462134260997584e-04,
        # "w": 1400,
        # "h": 1400,
        "camera_model": "MEI",
    }
}

INDOOR_SAMPLE_1 = {
    "config_file": "configs/test/dac_dinov3l+dpt_indoor_test_m3d.json",
    "dataset_name": "matterport3d",
    "image_filename": "demo/input/matterport3d_rgb.png",
    "annotation_filename_depth": "demo/input/matterport3d_depth.exr",
    "depth_scale": 1.0,
    "fishey_grid": None,
    "fwd_sz": (512, 1024), # the patch size input to the model
    "erp": True,
    "cam_params": {
        'dataset':'matterport3d',
        # "w": 1024,
        # "h": 512,
        "camera_model": "ERP",
    },
}

INDOOR_SAMPLE_2 = {
    "config_file": "configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json",
    "dataset_name": "scannetpp",
    "image_filename": "demo/input/scannetpp_rgb.jpg",
    "annotation_filename_depth": "demo/input/scannetpp_depth.png",
    "depth_scale": 1000.0,
    "fishey_grid": "demo/input/scannetpp_grid_fisheye.npy",
    "crop_wFoV": 180, # degree decided by origianl data fov + some buffer
    "fwd_sz": (512, 704), # the patch size input to the model
    "erp": False,
    "cam_params": {
        'dataset':'scannetpp',
        "fl_x": 789.9080967683176,
        "fl_y": 791.5566599926353,
        "cx": 879.203786509326,
        "cy": 584.7893145555763,
        "k1": -0.029473047856246333,
        "k2": -0.005769803970428537,
        "k3": -0.002148236771485755,
        "k4": 0.00014840568362061509,
        # "w": 1752,
        # "h": 1168,
        "camera_model": "OPENCV_FISHEYE",
    }
}

def demo_one_sample(model, device, sample, cano_sz, args: argparse.Namespace):
    #######################################################################
    ############# Prepare Data ##############
    #######################################################################
    
    image = np.asarray(
        Image.open(sample["image_filename"])
    )
    org_img_h, org_img_w = image.shape[:2]
    if sample["annotation_filename_depth"] is None:
        depth = np.zeros((org_img_h, org_img_w), dtype=np.float32)
    else:
        depth = (
            np.asarray(
                cv2.imread(sample["annotation_filename_depth"], cv2.IMREAD_ANYDEPTH)
            ).astype(np.float32)
            / sample["depth_scale"]
        )

    dataset_name = sample["dataset_name"]
    fwd_sz=sample["fwd_sz"]

    if not sample["erp"]:
        # convert depth from zbuffer to euclid 
        if dataset_name in ['nyu', 'kitti']:
            x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
            depth = depth * np.sqrt((x - sample["cam_params"]['cx'])**2 + (y - sample["cam_params"]['cy'])**2 + sample["cam_params"]['fx']**2) / sample["cam_params"]['fx']
            depth = depth.astype(np.float32)
        elif dataset_name == 'scannetpp': # Very critical for scannet++ fisheye. Skip kitti360 because we prepared the depth already in euclid.
            # For fisheye, converting back to euclid with undistorted ray direction via the ray lookup table for efficiency
            fisheye_grid = np.load(sample["fishey_grid"])
            fisheye_grid_z = cv2.resize(fisheye_grid[:, :, 2], (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
            depth = depth / fisheye_grid_z
        depth = depth.astype(np.float32)
        phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0

        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > 0.01
                
        # Automatically calculate the erp crop size
        crop_width = int(cano_sz[0] * sample["crop_wFoV"] / 180)
        crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])
        
        # convert to ERP
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            crop_height, crop_width, cano_sz[0], cano_sz[0]*2, sample["cam_params"], roll, scale_fac=None
        )
        lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])
            
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask, lat_grid, long_grid = resize_for_input((image * 255.).astype(np.uint8), depth, fwd_sz, None, [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask, lat_grid=latitude, long_grid=longitude)
    else:
        attn_mask = np.ones_like(depth)
        lat_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
        long_range = torch.tensor([-np.pi, np.pi], dtype=torch.float32)
        H, W = image.shape[:2]
        latitude, longitude = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, H), np.linspace(-np.pi, np.pi, W), indexing='ij')
        # resizing process to fwd_sz.
        to_cano_ratio = cano_sz[0] / image.shape[0]
        image, depth, pad, pred_scale_factor, lat_grid, long_grid = resize_for_input(image, depth, fwd_sz, None, cano_sz, to_cano_ratio, lat_grid=latitude, long_grid=longitude)


    # convert to tensor batch
    normalization_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    image = TF.normalize(TF.to_tensor(image), **normalization_stats)
    gt = TF.to_tensor(depth)
    mask = TF.to_tensor((depth > 0.01).astype(np.uint8))
    attn_mask = TF.to_tensor((attn_mask>0).astype(np.float32))
    batch = {
        "image": image.unsqueeze(0),
        "gt": gt.unsqueeze(0),
        "mask": mask.unsqueeze(0),
        "attn_mask": attn_mask.unsqueeze(0),
        "lat_range": lat_range.unsqueeze(0),
        "long_range": long_range.unsqueeze(0),
        "lat_grid": torch.tensor(lat_grid).unsqueeze(0),
        "long_grid": torch.tensor(long_grid).unsqueeze(0),
        "info": {
            "pred_scale_factor": pred_scale_factor,
        },
    }

    #######################################################################
    ########################### Model Inference ###########################
    #######################################################################

    gt, mask, attn_mask, lat_range, long_range, lat_grid, long_grid = batch["gt"].to(device), batch["mask"].to(device), batch['attn_mask'].to(device), batch["lat_range"].to(device), batch["long_range"].to(device), batch["lat_grid"].to(device), batch["long_grid"].to(device)
    with torch.no_grad():
        preds, _, _ = model(batch["image"].to(device), lat_range, long_range, attn_mask=attn_mask, lat_grid=lat_grid)
    preds *= pred_scale_factor

    #######################################################################
    ##################  Visualization and Output results  #################
    #######################################################################
    save_img_dir = os.path.join(args.out_dir)
    os.makedirs(save_img_dir, exist_ok=True)
    if 'attn_mask' in batch.keys():
        attn_mask = batch['attn_mask'][0]
    else:
        attn_mask = None

    # adjust vis_depth_max for outdoor datasets
    if dataset_name == 'kitti360':
        vis_depth_max = 40.0
        vis_arel_max = 0.3
    else:
        # default indoor visulization parameters
        vis_depth_max = 10.0
        vis_arel_max = 0.5

    rgb = save_val_imgs_v2(
        0,
        preds[0],
        batch["gt"][0],
        batch["image"][0],
        f'{dataset_name}_output.jpg',
        save_img_dir,
        active_mask=attn_mask,
        valid_depth_mask=batch["mask"][0],
        depth_max=vis_depth_max,
        arel_max=vis_arel_max
    )

def main(args: argparse.Namespace):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = None

    samples = [OUTDOOR_SAMPLE_1, INDOOR_SAMPLE_1, INDOOR_SAMPLE_2]
    for i, sample in enumerate(samples):
        
        config_file = sample["config_file"]
        with open(config_file, "r") as f:
            config = json.load(f)
        
        if model is None:
            model_name = 'UniDAC'
            model = eval(model_name).build(config)
            model.load_pretrained(args.model_file)
            model = model.to(device)
            model.eval()
            
        print(f"Processing sample {i}: {sample['dataset_name']}...")
        cano_sz = config["data"]["cano_sz"]
        demo_one_sample(model, device, sample, cano_sz, args)

    print('Demo completed!!')



if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    # parser.add_argument("--config-file", type=str, default="checkpoints/dac_swinl_outdoor.json")
    parser.add_argument("--model-file", type=str, default="checkpoints/unidac.pt")
    parser.add_argument("--out-dir", type=str, default='demo/output')
    
    args = parser.parse_args()

    main(args)