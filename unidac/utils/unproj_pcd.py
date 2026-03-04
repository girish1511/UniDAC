import numpy as np
import torch
# from plyfile import PlyData, PlyElement
import cv2


def get_pcd_base(H, W, u0, v0, fx, fy):
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32)
    u_m_u0 = x - u0

    y_col = np.arange(0, H)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (W, 1)).T
    y = y.astype(np.float32)
    v_m_v0 = y - v0

    x = u_m_u0 / fx
    y = v_m_v0 / fy
    z = np.ones_like(x)
    pw = np.stack([x, y, z], axis=2)  # [h, w, c]
    return pw


def reconstruct_pcd(depth, fx, fy, u0, v0, pcd_base=None, mask=None):
    if type(depth) == torch.__name__:
        depth = depth.cpu().numpy().squeeze()
    depth = cv2.medianBlur(depth, 5)
    if pcd_base is None:
        H, W = depth.shape
        pcd_base = get_pcd_base(H, W, u0, v0, fx, fy)
    pcd = depth[:, :, None] * pcd_base
    if mask is not None:
        pcd[mask] = 0
    return pcd

def reconstruct_pcd_erp(depth, mask=None, lat_range=None, long_range=None):
    "Assume depth in euclid distane"
    if type(depth) == torch.__name__:
        depth = depth.cpu().numpy().squeeze()
    depth = cv2.medianBlur(depth, 5)
    H, W = depth.shape
    # Assume to use hemishperes of 360 degree camera if no range is given
    if lat_range is None:
        latitude = np.linspace(-np.pi / 2, np.pi / 2, H)
        longitude = np.linspace(np.pi, 0, W)
    else:
        latitude = np.linspace(lat_range[0], lat_range[1], H)
        longitude = np.linspace(long_range[0], long_range[1], W)
    
    longitude, latitude = np.meshgrid(longitude, latitude)
    longitude = longitude.astype(np.float32)
    latitude = latitude.astype(np.float32)

    x = -np.cos(latitude) * np.cos(longitude)
    z = np.cos(latitude) * np.sin(longitude)
    y = np.sin(latitude)
    
    # This does not work for large FOV cameras
    # z_mask = z < 0
    # x = np.where(z_mask, 1/np.sqrt(3), x)
    # y = np.where(z_mask, 1/np.sqrt(3), y)
    # z = np.where(z_mask, 1/np.sqrt(3), z)
    
    pcd_base = np.concatenate([x[:, :, None], y[:, :, None], z[:, :, None]], axis=2)
    pcd = depth[:, :, None] * pcd_base
    # if mask is not None:
    #     pcd[mask] = 0
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().squeeze().astype(bool)
        mask = mask.astype(bool)
        pcd[~mask] = 0
    return pcd

def reconstruct_pcd_fisheye(depth, grid_fisheye, pcd_base=None, mask=None):
    "assume the lookup table recose ray direction with norm 1, depth in euclid distance"
    if type(depth) == torch.__name__:
        depth = depth.cpu().numpy().squeeze()
    depth = cv2.medianBlur(depth, 5)
    if pcd_base is None:
        H, W = depth.shape
        pcd_base = grid_fisheye[:, :, :3]
        # # if depth is in z-buffer, need to normalize z to be 1 before apply depth (edge grids might go to infinity)
        # pcd_base /= pcd_base[:, :, 2][:, :, None]
    pcd = depth[:, :, None] * pcd_base
    if mask is not None:
        pcd[mask>0, :] = 0
    return pcd
