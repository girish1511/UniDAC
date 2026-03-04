from typing import Any, Dict, List, Tuple

import matplotlib.cm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os, cv2
import torch
from dac.utils.erp_geometry import erp_patch_to_cam_fast
from dac.utils.unproj_pcd import reconstruct_pcd, reconstruct_pcd_fisheye, reconstruct_pcd_erp


def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    if value.ndim > 2:
        return value
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[..., :3]
    return img


def image_grid(imgs: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
    if not len(imgs):
        return None

    assert len(imgs) == rows * cols
    h, w = imgs[0].shape[:2]
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(
            Image.fromarray(img.astype(np.uint8)).resize((w, h), Image.ANTIALIAS),
            box=(i % cols * w, i // cols * h),
        )

    return np.array(grid)


def get_pointcloud_from_rgbd(
    image: np.array,
    depth: np.array,
    mask: np.ndarray,
    intrinsic_matrix: np.array,
    extrinsic_matrix: np.array = None,
):
    depth = np.array(depth).squeeze()
    mask = np.array(mask).squeeze()
    # Mask the depth array
    masked_depth = np.ma.masked_where(mask == False, depth)
    # masked_depth = np.ma.masked_greater(masked_depth, 8000)
    # Create idx array
    idxs = np.indices(masked_depth.shape)
    u_idxs = idxs[1]
    v_idxs = idxs[0]
    # Get only non-masked depth and idxs
    z = masked_depth[~masked_depth.mask]
    compressed_u_idxs = u_idxs[~masked_depth.mask]
    compressed_v_idxs = v_idxs[~masked_depth.mask]
    image = np.stack(
        [image[..., i][~masked_depth.mask] for i in range(image.shape[-1])], axis=-1
    )

    # Calculate local position of each point
    # Apply vectorized math to depth using compressed arrays
    cx = intrinsic_matrix[0, 2]
    fx = intrinsic_matrix[0, 0]
    x = (compressed_u_idxs - cx) * z / fx
    cy = intrinsic_matrix[1, 2]
    fy = intrinsic_matrix[1, 1]
    # Flip y as we want +y pointing up not down
    y = -((compressed_v_idxs - cy) * z / fy)

    # # Apply camera_matrix to pointcloud as to get the pointcloud in world coords
    # if extrinsic_matrix is not None:
    #     # Calculate camera pose from extrinsic matrix
    #     camera_matrix = np.linalg.inv(extrinsic_matrix)
    #     # Create homogenous array of vectors by adding 4th entry of 1
    #     # At the same time flip z as for eye space the camera is looking down the -z axis
    #     w = np.ones(z.shape)
    #     x_y_z_eye_hom = np.vstack((x, y, -z, w))
    #     # Transform the points from eye space to world space
    #     x_y_z_world = np.dot(camera_matrix, x_y_z_eye_hom)[:3]
    #     return x_y_z_world.T
    # else:
    x_y_z_local = np.stack((x, y, z), axis=-1)
    return np.concatenate([x_y_z_local, image], axis=-1)


def save_file_ply(xyz, rgb, pc_file):
    if rgb.max() < 1.001:
        rgb = rgb * 255.0
    rgb = rgb.astype(np.uint8)
    # print(rgb)
    with open(pc_file, "w") as f:
        # headers
        f.writelines(
            [
                "ply\n" "format ascii 1.0\n",
                "element vertex {}\n".format(xyz.shape[0]),
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property uchar red\n",
                "property uchar green\n",
                "property uchar blue\n",
                "end_header\n",
            ]
        )

        for i in range(xyz.shape[0]):
            str_v = "{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n".format(
                xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2]
            )
            f.write(str_v)


# visualization code copied from Metric3D
def gray_to_colormap(img, cmap='rainbow'):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    img = img / (img.max() + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap


def get_data_for_log(pred: torch.tensor, target: torch.tensor, rgb: torch.tensor, active_mask: torch.tensor=None, valid_depth_mask: torch.tensor=None, depth_max=80):
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()
    rgb = ((rgb * std) + mean).astype(np.uint8)
    if active_mask is not None:
        active_mask = active_mask.squeeze().cpu().numpy()
        rgb = rgb * (active_mask>0).astype(np.uint8)
        pred = pred * (active_mask>0).astype(np.float32)
    if valid_depth_mask is not None:
        valid_depth_mask = valid_depth_mask.squeeze().cpu().bool().numpy()
        target = target * valid_depth_mask

    pred[pred<0] = 0
    target[target<0] = 0
    # # mask out pred borders
    # border_mask = np.logical_and(rgb[0] == 0, rgb[1] == 0, rgb[2] == 0)
    # pred[border_mask] = 0  
    if depth_max is not None:
        pred[pred>depth_max] = depth_max
        target[target>depth_max] = depth_max
        max_scale = depth_max
    else:
        max_scale = max(pred.max(), target.max())
    pred_scale = (pred/max_scale * 10000).astype(np.uint16)
    target_scale = (target/max_scale * 10000).astype(np.uint16)
    pred_color = gray_to_colormap(pred)
    target_color = gray_to_colormap(target)
    pred_color = cv2.resize(pred_color, (rgb.shape[2], rgb.shape[1]))
    target_color = cv2.resize(target_color, (rgb.shape[2], rgb.shape[1]))

    return rgb, pred_scale, target_scale, pred_color, target_color


def save_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    target: torch.tensor,
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    tb_logger=None,
    active_mask: torch.tensor=None,
    valid_depth_mask: torch.tensor=None
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    rgb, pred_scale, target_scale, pred_color, target_color = get_data_for_log(pred, target, rgb, active_mask=active_mask, valid_depth_mask=valid_depth_mask)
    rgb = rgb.transpose((1, 2, 0))
    cat_img = np.concatenate([rgb, pred_color, target_color], axis=0)
    plt.imsave(os.path.join(save_dir, filename[:-4]+'_subplot.jpg'), cat_img)

    # save to tensorboard
    if tb_logger is not None:
        tb_logger.add_image(f'{filename[:-4]}_subplot.jpg', cat_img.transpose((2, 0, 1)), iter)
    return rgb


def save_val_imgs_v2(
    iter: int, 
    depth_pred: torch.tensor, 
    depth_gt: torch.tensor,
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    active_mask: torch.tensor=None,
    valid_depth_mask: torch.tensor=None,
    depth_max=20,
    arel_max=0.3,
    save_sep_imgs=False
    ):
    
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    depth_pred = depth_pred.squeeze().cpu().numpy()
    depth_gt = depth_gt.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()
    rgb = ((rgb * std) + mean).astype(np.uint8)
    if active_mask is not None:
        active_mask = active_mask.squeeze().cpu().numpy()
        rgb = rgb * (active_mask>0).astype(np.uint8)
        depth_pred = depth_pred * (active_mask>0).astype(np.float32)
    if valid_depth_mask is not None:
        valid_depth_mask = valid_depth_mask.squeeze().cpu().bool().numpy()
        depth_gt = depth_gt * valid_depth_mask
    rgb = rgb.transpose((1, 2, 0))
    
    # compute error, you have zero divison where depth_gt == 0.0
    depth_arel = np.abs(depth_gt - depth_pred) / (depth_gt + 1e-9)
    depth_arel[depth_gt == 0.0] = 0.0
    
    cmap_depth = cm.magma_r  # Use any colormap you like (e.g., 'viridis', 'plasma', 'inferno', etc.)
    norm_depth = mcolors.Normalize(vmin=0, vmax=depth_max)  # Set the data range for the color bar
    cmap_arel = cm.coolwarm  # Use any colormap you like (e.g., 'viridis', 'plasma', 'inferno', etc.)
    norm_arel = mcolors.Normalize(vmin=0, vmax=arel_max)  # Set the data range for the color bar

    if save_sep_imgs:
        # save all the subplots as individual images
        plt.figure()
        plt.imshow(rgb)
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        # plt.title("Image")
        plt.savefig(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), dpi=200, bbox_inches='tight', pad_inches=0)
        
        plt.figure()
        plt.imshow(depth_arel, cmap=cmap_arel, norm=norm_arel)
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        # No need to save colorbar as it is saved in the main image
        # plt.colorbar(ax, label="A.Rel")
        # plt.title("A. Rel")
        plt.savefig(os.path.join(save_dir, filename[:-4]+'_arel.jpg'), dpi=200, bbox_inches='tight', pad_inches=0)
        
        plt.figure()
        ax = plt.imshow(depth_gt, cmap=cmap_depth, norm=norm_depth)
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        plt.savefig(os.path.join(save_dir, filename[:-4]+'_gt.jpg'), dpi=200, bbox_inches='tight', pad_inches=0)
        
        plt.figure()
        ax = plt.imshow(depth_pred, cmap=cmap_depth, norm=norm_depth)
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        plt.savefig(os.path.join(save_dir, filename[:-4]+'_pred.jpg'), dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close('all')
    else:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(rgb)
        plt.title("Image")
        
        plt.subplot(2, 2, 2)
        ax=plt.imshow(depth_arel, cmap=cmap_arel, norm=norm_arel)
        plt.colorbar(ax, label="A.Rel")
        plt.title("A. Rel")
        
        plt.subplot(2, 2, 3)
        ax = plt.imshow(depth_gt, cmap=cmap_depth, norm=norm_depth)
        plt.colorbar(ax, label="Meter")
        plt.title("Depth GT")
        
        plt.subplot(2, 2, 4)
        ax = plt.imshow(depth_pred, cmap=cmap_depth, norm=norm_depth)
        plt.colorbar(ax, label="Meter")
        plt.title("Depth Pred")
        
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(os.path.join(save_dir, filename[:-4]+'_subplot.jpg'), dpi=200)
        # plt.close()
    return rgb

def save_val_imgs_v3(
    iter: int, 
    depth_pred: torch.tensor, 
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    active_mask: torch.tensor=None,
    valid_depth_mask: torch.tensor=None,
    depth_max=20,
    ):
    """
       Conbine predicted depth and rgb in a single image to save.
    """
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    depth_pred = depth_pred.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()
    rgb = ((rgb * std) + mean).astype(np.uint8)
    if active_mask is not None:
        active_mask = active_mask.squeeze().cpu().numpy()
        rgb = rgb * (active_mask>0).astype(np.uint8)
        depth_pred = depth_pred * (active_mask>0).astype(np.float32)
    if valid_depth_mask is not None:
        valid_depth_mask = valid_depth_mask.squeeze().cpu().bool().numpy()
    rgb = rgb.transpose((1, 2, 0))
    pred_vis = colorize(depth_pred, vmin=0, vmax=depth_max)
    combined_img = np.concatenate((rgb, pred_vis), axis=1)
    plt.imsave(os.path.join(save_dir, filename), combined_img)
    return combined_img
        
def visualize_results(batch, preds, out_dir, config, data_dir, save_pcd=False, index=0):
    save_img_dir = os.path.join(out_dir, 'val_imgs')
    dataset_name = config['data']['data_root']
    os.makedirs(save_img_dir, exist_ok=True)
    if 'attn_mask' in batch.keys():
        attn_mask = batch['attn_mask'][0]
    else:
        attn_mask = None

    # adjust vis_depth_max for outdoor datasets
    if dataset_name == 'kitti360':
        vis_depth_max = 40.0
        vis_arel_max = 0.3
    elif dataset_name == 'kitti':
        vis_depth_max = 80.0
        vis_arel_max = 0.3
    else:
        # default indoor visulization parameters
        vis_depth_max = 10.0
        vis_arel_max = 0.5

    rgb = save_val_imgs_v2(
        index,
        preds[0],
        batch["gt"][0],
        batch["image"][0],
        f'{index:06d}.jpg',
        save_img_dir,
        active_mask=attn_mask,
        valid_depth_mask=batch["mask"][0],
        depth_max=vis_depth_max,
        arel_max=vis_arel_max
    )
    
    intrinsics = batch['info']['camera_intrinsics'][0].detach().cpu().numpy()
    if save_pcd:
        if config['data']['tgt_f'] > 0 and config['data']['erp'] == False: # perspective model output
            pcd = reconstruct_pcd(preds[0, 0].detach().cpu().numpy(), intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        else:
            pcd = reconstruct_pcd_erp(preds[0, 0].detach().cpu().numpy(), mask=(batch['attn_mask'][0][0]).numpy(), lat_range=batch['lat_range'][0], long_range=batch['long_range'][0]) 
        save_pcd_dir = os.path.join(out_dir, 'val_pcds')
        os.makedirs(os.path.join(save_pcd_dir), exist_ok=True)
        pc_file = os.path.join(save_pcd_dir, f'pcd_{index:06d}.ply')
        pcd = pcd.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)
        # if dataset_name not in ['matterport3d', 'gibson_v2']:
        #     non_zero_indices = pcd[:, -1] > 0
        #     pcd = pcd[non_zero_indices]
        #     rgb = rgb[non_zero_indices]
        save_file_ply(pcd, rgb, pc_file)
    
    
    ##########  Additioanal step for erp mode: converting the testing erp image back to original space for visualization  ##########
    if 'erp' in config['data'].keys() and config['data']['erp'] == True:                    
        if dataset_name == 'kitti360':
            out_h = 700
            out_w = 700
            fisheye_file = batch['info']['image_filename'][0]
            if 'image_02' in fisheye_file:
                grid_fisheye = np.load(os.path.join('splits', 'kitti360', 'grid_fisheye_02.npy'))
                mask_fisheye = np.load(os.path.join('splits', 'kitti360', 'mask_left_fisheye.npy'))
            elif 'image_03' in fisheye_file:
                grid_fisheye = np.load(os.path.join('splits', 'kitti360', 'grid_fisheye_03.npy'))
                mask_fisheye = np.load(os.path.join('splits', 'kitti360', 'mask_right_fisheye.npy'))
            grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            mask_fisheye = cv2.resize(mask_fisheye.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            cam_params={'dataset':'kitti360'}
            phi = 0.
            
        elif dataset_name == 'scannetpp':
            """
                Currently work perfect with phi = 0. For larger phi, corners may have artifacts.
            """
            grid_fisheye = np.load(os.path.join(data_dir, 'data', batch['info']['scene_id'][0], 'dslr', 'grid_fisheye.npy'))
            # set output size the same aspact ratio as raw image (no need to be same as fw_size)
            out_h = 500
            out_w = 750
            grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            cam_params={'dataset':'scannetpp'} # when grid table is available, no need for intrinsic parameters
            phi = batch['info']['phi'][0].detach().cpu().numpy()

        elif dataset_name == 'nyu':  
            # set output size the same aspact ratio as raw image (no need to be same as fw_size)
            s_ratio = 1.0
            out_h = int(480 * s_ratio)
            out_w = int(640 * s_ratio)
            grid_fisheye = None
            # uncomment if treating nyu as fisheye camera, with slight distortion
            # grid_fisheye = np.load('splits/nyu/grid_fisheye.npy')
            # grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            # grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            # grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            cam_params={'dataset':'nyu', 'fx':intrinsics[0, 0]* s_ratio, 'fy':intrinsics[1, 1]* s_ratio, 'cx':intrinsics[0, 2]* s_ratio, 'cy':intrinsics[1, 2]* s_ratio} # when grid table is available, no need for intrinsic parameters
            phi = batch['info']['phi'][0].detach().cpu().numpy()

        elif dataset_name == 'kitti':        
            # Set output size the same aspact ratio as raw image (no need to be same as fw_size)
            # Image size and intrinsic parameters need to be changed together make the output conversion right (no depth scale change involved)
            s_ratio = 1.0
            out_h = int(375 * s_ratio)
            out_w = int(1242 * s_ratio)
            grid_fisheye = None
            cam_params={'dataset':'kitti', 'fx':intrinsics[0, 0]* s_ratio, 'fy':intrinsics[1, 1]* s_ratio, 'cx':intrinsics[0, 2]* s_ratio, 'cy':intrinsics[1, 2]* s_ratio}
            phi = 0.
            
        # convert the ERP result back to camera space for visualization (No need for original ERP image)
        if dataset_name not in ['matterport3d', 'gibson_v2']:    
            # scale the full erp_size depth scaling factor is equivalent to resizing data (given same aspect ratio)
            erp_h = config['data']['cano_sz'][0]
            erp_h = erp_h * batch['info']['pred_scale_factor'][0].detach().cpu().numpy()
            if 'f_align_factor' in batch['info']:
                erp_h = erp_h / batch['info']['f_align_factor'][0].detach().cpu().numpy()
            img_out, depth_out, valid_mask, active_mask, depth_out_gt = erp_patch_to_cam_fast(
                batch["image"][0], preds[0].detach().cpu(), attn_mask, 0., phi, out_h=out_h, out_w=out_w, erp_h=erp_h, erp_w=erp_h*2, cam_params=cam_params, 
                fisheye_grid2ray=grid_fisheye, depth_erp_gt=batch["gt"][0].detach().cpu())
            rgb = save_val_imgs_v2(
                index,
                depth_out,
                depth_out_gt,
                img_out,
                f'{index:06d}_remap.jpg',
                save_img_dir,
                active_mask=active_mask,
                depth_max=vis_depth_max,
                arel_max=vis_arel_max
                )