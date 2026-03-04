"""
Core Geometry Functions introduced in DepthAnyCamera framework
"""

import os
import sys
import cv2
import math
import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F


def prepare_erp_deform_cnn_offsets(img_h, img_w, lat_range=(-np.pi/2, np.pi/2), long_range=(-np.pi, np.pi), kernel_size=3):
    """
    Prepare the offsets for the deformable convolutional layer
    - Implement using explicit Gnomonic Projection -- https://mathworld.wolfram.com/GnomonicProjection.html
    
    Each kernel at different locations will have different offsets, which are computed based on the Gnomonic Projection.
    A regular kernel can be thought of tangent image centered at each location
    """
    lat_step = lat_range[1] - lat_range[0]
    long_step = long_range[1] - long_range[0]
    lat_res = lat_step / img_h
    long_res = long_step / img_w
    
    # np.meshgrid defaut indexing is 'xy', which is different from 'ij' in torch
    lat_grid, long_grid = np.meshgrid(
        np.linspace(lat_range[0] + lat_res/2, lat_range[1] - lat_res/2, img_h),
        np.linspace(long_range[0] + long_res/2, long_range[1] - long_res/2, img_w),
        indexing='ij'
    )
    lat_grid = lat_grid.reshape(1, 1, img_h, img_w)
    long_grid = long_grid.reshape(1, 1, img_h, img_w)

    # np.meshgrid defaut indexing is 'xy', which is different from 'ij' in torch
    # kernel size is usually the number of grids, so the length is kernel_size-1
    kernel_y_vec, kernel_x_vec = np.meshgrid(
        np.linspace(-1, 1, kernel_size) * (kernel_size-1) / 2,
        np.linspace(-1, 1, kernel_size) * (kernel_size-1) / 2,
        indexing='ij'
    )
    kernel_x_vec = kernel_x_vec.reshape(1, kernel_size*kernel_size, 1, 1)
    kernel_y_vec = kernel_y_vec.reshape(1, kernel_size*kernel_size, 1, 1)

    # TODO: convert pixel coordiantes to canonical coordinates with radio = 1 or f = 1 (why not reflect the distortion well)
    # calculate the canonical f the the limit at the equitor
    # cano_f = 1 / np.tan(lat_step / img_h)
    kernel_x_vec_cano = kernel_x_vec #/ cano_f
    kernel_y_vec_cano = kernel_y_vec #/ cano_f
    
    rho = np.sqrt(kernel_x_vec_cano**2 + kernel_y_vec_cano**2)
    c = np.arctan(rho)
    
    # The range is between -pi/2 and pi/2 due to arcsin function
    kernel_lat_grid = np.arcsin(np.cos(c) * np.sin(lat_grid) + kernel_y_vec_cano * np.sin(c) * np.cos(lat_grid) / (rho+1e-9))
    # The simpling dealing with nan values OK?
    kernel_long_grid = long_grid + np.arctan(kernel_x_vec_cano * np.sin(c) / (rho * np.cos(lat_grid) * np.cos(c) - kernel_y_vec_cano * np.sin(lat_grid) * np.sin(c) + 1e-9))
    
    # convert lat and long to erp x and y
    kernel_x_grid_local = kernel_long_grid / long_res - long_grid / long_res
    kernel_y_grid_local = kernel_lat_grid / lat_res - lat_grid / lat_res
    
    # TODO: Use the (long=0, lat=0) kernel size for normalization? Compute the ratio to the input kernel size
    kerne_x_grid_center = kernel_x_grid_local[0, :, img_h//2, img_w//2]
    kerne_y_grid_center = kernel_y_grid_local[0, :, img_h//2, img_w//2]
    x_factor = (kernel_size-1) / (kerne_x_grid_center.max() - kerne_x_grid_center.min())
    y_factor = (kernel_size-1) / (kerne_y_grid_center.max() - kerne_y_grid_center.min())
    
    # compute the offsets in pixels units
    offsets = np.zeros((2, kernel_size*kernel_size, img_h, img_w), dtype=np.float32)
    offsets[0] = kernel_x_grid_local - kerne_x_grid_center.reshape(1, kernel_size*kernel_size, 1, 1)
    offsets[1] = kernel_y_grid_local - kerne_y_grid_center.reshape(1, kernel_size*kernel_size, 1, 1)
    erp_kernel_grid_local = np.concatenate([kernel_x_grid_local, kernel_y_grid_local], axis=0)
    
    offsets[0] *= x_factor
    offsets[1] *= y_factor
    erp_kernel_grid_local[0] *= x_factor
    erp_kernel_grid_local[1] *= y_factor
    return offsets, erp_kernel_grid_local


def cam_to_erp_patch_fast(img, depth, mask_valid_depth, theta, phi, patch_h, patch_w, erp_h, erp_w, cam_params, roll=None, scale_fac=None, padding_rgb=[123.675, 116.28, 103.53]):
    """
        This is an efficient implementation in two folds:
            - Only consider coordinates within target ERP patch
            - Implement using explicit Gnomonic Projection -- https://mathworld.wolfram.com/GnomonicProjection.html
            
        
        Args:
            img: the source perspective image [img_h, img_w, 3]
            depth: the corresponding depth map [img_h, img_w, 1]
            mask_valid_depth: the valid depth mask [img_h, img_w, 1]
            theta: the longitude of the target patch center
            phi: the latitude of the target patch center
            patch_h: the height of the target patch
            patch_w: the width of the target patch
            erp_h: the height of the whole equirectangular projection
            erp_w: the width of the whole equirectangular projection
            cam_params: the camera parameters, check the usage in code for details
            roll: the camera roll angle in radians
            scale_aug: the scale augmentation factor, 0 means no augmentation
        output:
            erp_img: the target patch in equirectangular projection [3, patch_h, patch_w]
            erp_depth: the corresponding depth in equirectangular projection [patch_h, patch_w]
            erp_mask_valid_depth: the valid depth mask in equirectangular projection [patch_h, patch_w]
            mask_active: the mask indicating the valid area in the target patch [patch_h, patch_w]
            lat_grid: the latitude grid in the target patch [patch_h, patch_w]
            lon_grid: the longitude grid in the target patch [patch_h, patch_w]
    """
    [img_h, img_w, _] = img.shape

    img_new = np.transpose(img, [2, 0, 1])
    img_new = torch.from_numpy(img_new).unsqueeze(0)
    depth_new = np.transpose(depth, [2, 0, 1])
    depth_new = torch.from_numpy(depth_new).unsqueeze(0)
    mask_valid_depth = np.transpose(mask_valid_depth, [2, 0, 1])
    mask_valid_depth = torch.from_numpy(mask_valid_depth).unsqueeze(0)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    # compute the target FOV based on target patch size and whole erp size
    wFOV_tgt = patch_w / erp_w * PI2
    hFOV_tgt = patch_h / erp_h * PI

    # only target patch erp coordinates
    cp = torch.tensor([theta, phi]).view(1, 1, -1)
    lat_grid, lon_grid = torch.meshgrid(
        torch.linspace(phi - hFOV_tgt/2, phi + hFOV_tgt/2, patch_h),
        torch.linspace(theta - wFOV_tgt/2, theta + wFOV_tgt/2, patch_w))
    lon_grid = lon_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
    lat_grid = lat_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
        
    # TODO: lat_grid may need cap to -pi/2 and pi/2 if crop size is large and pitch angle is large
    # compute corresponding perp image coordinates via Gnomonic Project explicitly (x, y given sphere radius 1 or f=1 perspective image)
    cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(
        lon_grid - cp[..., 0])
    x_num = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0]))
    y_num = (torch.cos(cp[..., 1]) * torch.sin(lat_grid) - torch.sin(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(
        lon_grid - cp[..., 0]))
    new_x = x_num / cos_c
    new_y = y_num / cos_c

    # OPTIONAL: apply camera roll correction
    if roll is not None:
        roll = torch.tensor(roll, dtype=torch.float32)
        new_x_tmp = new_x * torch.cos(roll) - new_y * torch.sin(roll)
        new_y_tmp = new_x * torch.sin(roll) + new_y * torch.cos(roll)
        new_x = new_x_tmp
        new_y = new_y_tmp

    # Scale augmentation can just modify new_x and new_y directly, with depth factor
    if scale_fac is not None:
        new_x *= scale_fac
        new_y *= scale_fac
        depth_new *= scale_fac # depth value is adjusted for the scale augmentation
        # print(scale_fac)

    # Important: this normalization needs to use the source perspective image FOV, for the grid sample function range [-1, 1]
    # Gnomonic Projection for OPENCV_FISHEYE model, only works for FOV < 180 degree
    if 'camera_model' in cam_params.keys() and cam_params['camera_model'] == 'OPENCV_FISHEYE':    
        """
            Apply opencv distortion (Refer to: OpenCV)
        """
        k1 = cam_params['k1']
        k2 = cam_params['k2']
        k3 = cam_params['k3']
        k4 = cam_params['k4']
        fx = cam_params['fl_x']
        fy = cam_params['fl_y']
        cx = cam_params['cx']
        cy = cam_params['cy']

        # Option 1: original opencv fisheye distortion can not handle FOV >=180 degree or cos_c <= 0
        # r = np.sqrt(new_x*new_x + new_y*new_y)
        # theta = np.arctan(r)
        # theta_d = theta * (1 + k1*theta*theta + k2*theta**4 + k3*theta**6 + k4*theta**8)        
        # x_d = theta_d * new_x / (r+1e-9)
        # y_d = theta_d * new_y / (r+1e-9)

        # Option 2: A more numerically stable version able to handle FOV >=180 degree, adapted for Gnomonic Projection
        r = np.sqrt(x_num*x_num + y_num*y_num)
        theta = np.arccos(cos_c)
        theta_d = theta * (1 + k1*theta*theta + k2*theta**4 + k3*theta**6 + k4*theta**8)
        x_d = theta_d * x_num / (r)
        y_d = theta_d * y_num / (r)
        
        # project to image coordinates
        new_x = fx * x_d + cx
        new_y = fy * y_d + cy
        
        """
            Projection to image coordinates using intrinsic parameters
        """
        new_x -= img_w/2
        new_x /= (img_w/2)
        new_y -= img_h/2
        new_y /= (img_h/2)
        
    # Gnomonic Projection for MEI model, but only works for FOV < 180 degree (kitti360 is slightly beyond 180)
    elif 'camera_model' in cam_params.keys() and cam_params['camera_model'] == 'MEI':
        xi = cam_params['xi']
        k1 = cam_params['k1']
        k2 = cam_params['k2']
        p1 = cam_params['p1']
        p2 = cam_params['p2']
        fx = cam_params['fx']
        fy = cam_params['fy']
        cx = cam_params['cx']
        cy = cam_params['cy']
        
        # Adpated for Gnomonic Projection
        p_u = x_num / (cos_c + xi)
        p_v = y_num / (cos_c + xi)

        # apply distortion
        ro2 = p_u*p_u + p_v*p_v

        p_u *= 1 + k1*ro2 + k2*ro2*ro2
        p_v *= 1 + k1*ro2 + k2*ro2*ro2

        p_u += 2*p1*p_u*p_v + p2*(ro2 + 2*p_u*p_u)
        p_v += p1*(ro2 + 2*p_v*p_v) + 2*p2*p_u*p_v

        # apply projection
        new_x = fx*p_u + cx 
        new_y = fy*p_v + cy
        
        """
            Projection to image coordinates using intrinsic parameters
        """
        new_x -= img_w/2
        new_x /= (img_w/2)
        new_y -= img_h/2
        new_y /= (img_h/2)
        
    # elif cam_params['dataset'] == 'nyu': # uncomment if you want to consider nyu as fisheye, very slight distortion
    #     kc = [cam_params['k1'], cam_params['k2'], cam_params['p1'], cam_params['p2'], cam_params['k3']]
    #     fx = cam_params['fx']
    #     fy = cam_params['fy']
    #     cx = cam_params['cx']
    #     cy = cam_params['cy']
    #
    #     """
    #         Apply distortion (Refer to: NYUv2 Toolbox and http://www.vision.caltech.edu/bouguetj/calib_doc/)
    #     """
    #     r_sq = new_x**2 + new_y**2
    #     dx = 2*kc[2]*new_x*new_y + kc[3]*(r_sq + 2*new_x**2)
    #     dy = kc[2]*(r_sq + 2*new_y**2) + 2*kc[3]*new_x*new_y
    #     new_x = (1 + kc[0]*r_sq + kc[1]*r_sq**2 + kc[4]*r_sq**3)*new_x + dx
    #     new_y = (1 + kc[0]*r_sq + kc[1]*r_sq**2 + kc[4]*r_sq**3)*new_y + dy

    #     """
    #         Projection to image coordinates using intrinsic parameters
    #     """
    #     new_x = fx * new_x + cx
    #     new_y = fy * new_y + cy

    #     # convert to grid_sample range [-1, 1] scope (could extend due to larger ERP range or shifted principle center)
    #     new_x -= img_w/2
    #     new_x /= (img_w/2)
    #     new_y -= img_h/2
    #     new_y /= (img_h/2)
    else:
        # If necessuary, handle principal point shift in perspective data (e.g., KITTI, DDAD, LYFT)
        if 'cx' in cam_params.keys():
            new_x = cam_params['fx'] * new_x + cam_params['cx']
            new_y = cam_params['fy'] * new_y + cam_params['cy']
            # convert to grid_sample range [-1, 1] scope (could extend due to larger ERP range or shifted principle center)
            new_x -= img_w/2
            new_x /= (img_w/2)
            new_y -= img_h/2
            new_y /= (img_h/2)
        else:    
            # assume FOV in radians
            new_x = new_x / np.tan(cam_params['wFOV'] / 2)
            new_y = new_y / np.tan(cam_params['hFOV'] / 2)

    new_x = new_x.reshape(1, patch_h, patch_w)
    new_y = new_y.reshape(1, patch_h, patch_w)
    new_grid = torch.stack([new_x, new_y], -1)

    # those value within -1, 1 corresponding to content area
    mask_active = torch.logical_and(
        torch.logical_and(new_x > -1, new_x < 1),
        torch.logical_and(new_y > -1, new_y < 1),
    )*1.0

    # inverse mapping through grid_sample function in pytorch. Alternative is cv2.remap
    erp_img = F.grid_sample(img_new, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
    erp_img *= mask_active

    # compute depth in erp
    erp_depth = F.grid_sample(depth_new, new_grid, mode='nearest', padding_mode='border', align_corners=True)
    erp_depth *= mask_active

    # compute the valid depth mask in erp
    erp_mask_valid_depth = F.grid_sample(mask_valid_depth, new_grid, mode='nearest', padding_mode='border', align_corners=True)
    erp_mask_valid_depth *= mask_active

    # output
    erp_img = erp_img[0].permute(1, 2, 0).numpy()
    erp_depth = erp_depth[0, 0].numpy().astype(np.float32)
    erp_mask_valid_depth = erp_mask_valid_depth[0, 0].numpy().astype(np.float32)
    mask_active = mask_active[0].numpy().astype(np.float32)
    lat_grid = lat_grid.reshape(patch_h, patch_w).numpy().astype(np.float32)
    lon_grid = lon_grid.reshape(patch_h, patch_w).numpy().astype(np.float32)
    
    # # apply invalid region to padding_rgb
    # erp_img[erp_mask_valid_depth == 0] = np.array(padding_rgb)/255
    # erp_depth[mask_active==0] = np.array(padding_rgb[0])/255
    
    return erp_img, erp_depth, erp_mask_valid_depth, mask_active, lat_grid, lon_grid


def erp_patch_to_cam_fast(img_erp, depth_erp, mask_valid_erp, theta, phi, out_h, out_w, erp_h, erp_w, cam_params, fisheye_grid2ray=None, depth_erp_gt=None):
    """
        This is an efficient implementation in two folds:
            - Only consider coordinates within target ERP patch
            - Implement using explicit Gnomonic Projection -- https://mathworld.wolfram.com/GnomonicProjection.html
            
        
        Args:
            img_erp: the source perspective image [3, img_h, img_w] torch tensor
            depth_erp: the corresponding depth map [1, img_h, img_w] torch tensor
            mask_valid_erp: the valid depth mask [1, img_h, img_w] torch tensor
            theta: the longitude of the target patch center
            phi: the latitude of the target patch center
            out_h: the height of the output image
            out_w: the width of the output image
            erp_h: the height of the whole equirectangular projection
            erp_w: the width of the whole equirectangular projection
            cam_params: the camera parameters, check the usage in code for details
            fisheye_grid2ray: for models without closed-form inversion of undistortion, a lookup table can make the process faster (e.g. Mei Model from KITTI360)
        output:
            erp_img: the target patch in equirectangular projection [3, patch_h, patch_w]
            erp_depth: the corresponding depth in equirectangular projection [patch_h, patch_w]
            erp_mask_valid_depth: the valid depth mask in equirectangular projection [patch_h, patch_w]
            mask_active: the mask indicating the valid area in the target patch [patch_h, patch_w]
            lat_grid: the latitude grid in the target patch [patch_h, patch_w]
            lon_grid: the longitude grid in the target patch [patch_h, patch_w]
    """

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    theta = torch.tensor(theta, dtype=torch.float32)
    phi = torch.tensor(phi, dtype=torch.float32)

    img_y_grid, img_x_grid = torch.meshgrid(
        torch.linspace(0, out_h-1, out_h),
        torch.linspace(0, out_w-1, out_w))
    img_x_grid = img_x_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
    img_y_grid = img_y_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
    
    # convert img grid to normalized camera coordinates
    # if cam_params['dataset'] in ['kitti360', 'nyu', 'scannetpp']: # use this if you want to consider nyu as fisheye, very slight distortion
    if cam_params['dataset'] in ['kitti360', 'scannetpp', 'zipnerf']:
        """_
            MEI model has no closed-form inversion, so we need to provide the lookup table for fast inversion
        """
        X_new = torch.from_numpy(fisheye_grid2ray[:, :, 0]).reshape(1, -1)
        Y_new = torch.from_numpy(fisheye_grid2ray[:, :, 1]).reshape(1, -1)
        Z_new = torch.from_numpy(fisheye_grid2ray[:, :, 2]).reshape(1, -1)
        X_new = X_new / (Z_new + 1e-9)
        Y_new = Y_new / (Z_new + 1e-9)        
    else:
        if 'cx' in cam_params.keys():
            X_new = (img_x_grid - cam_params['cx']) / cam_params['fx']
            Y_new = (img_y_grid - cam_params['cy']) / cam_params['fy']
        else:
            img_x_grid -= out_w/2
            img_x_grid /= (out_w/2)
            img_y_grid -= out_h/2
            img_y_grid /= (out_h/2)
            # assume FOV in radians
            X_new = img_x_grid * np.tan(cam_params['wFOV'] / 2)
            Y_new = img_y_grid * np.tan(cam_params['hFOV'] / 2)
        
    # compute the corresponding latitude and longitude via Gnomonic Projection explicitly
    rho = torch.sqrt(X_new**2 + Y_new**2)
    c = torch.atan(rho)
    lat = torch.asin(torch.cos(c) * torch.sin(phi) + Y_new * torch.sin(c) * torch.cos(phi) / (rho+1e-9))
    lon = theta + torch.atan(X_new * torch.sin(c) / (rho * torch.cos(phi) * torch.cos(c) - Y_new * torch.sin(phi) * torch.sin(c) + 1e-9))
    
    # convert lat and long to erp x and y in normalized coordinates for grid_sample
    patch_h, patch_w = depth_erp.shape[-2:]
    lat_span = patch_h / erp_h * PI
    long_span = patch_w / erp_w * PI2
    erp_x_grid = (lon - theta) / long_span * 2
    erp_y_grid = (lat - phi) / lat_span * 2
    
    erp_x_grid = erp_x_grid.reshape(1, out_h, out_w)
    erp_y_grid = erp_y_grid.reshape(1, out_h, out_w)
    erp_grid = torch.stack([erp_x_grid, erp_y_grid], -1).to(torch.float32).to(img_erp.device)
    
    if fisheye_grid2ray is not None:
        isnan = torch.from_numpy(fisheye_grid2ray[:, :, 3]) # in lookup preparation, those invalid points are marked as nan
    else:
        isnan = torch.zeros(out_h, out_w)
    
    mask_active = torch.logical_and(
        torch.logical_and(erp_x_grid > -1, erp_x_grid < 1),
        torch.logical_and(erp_y_grid > -1, erp_y_grid < 1),
    )*1.0
    mask_active *= (isnan != 1).float()
    mask_active = mask_active.to(img_erp.device)

    # inverse mapping through grid_sample function in pytorch. Alternative is cv2.remap
    img_out = F.grid_sample(img_erp.unsqueeze(0), erp_grid, mode='bilinear', padding_mode='border', align_corners=True)
    depth_out = F.grid_sample(depth_erp.unsqueeze(0), erp_grid, mode='nearest', padding_mode='border', align_corners=True)
    mask_valid_out = F.grid_sample(mask_valid_erp.unsqueeze(0), erp_grid, mode='nearest', padding_mode='border', align_corners=True)
    img_out *= mask_active
    depth_out *= mask_active
    mask_valid_out *= mask_active
    
    # output
    # img_out = img_out[0].permute(1, 2, 0).numpy()
    # depth_out = depth_out[0, 0].numpy().astype(np.float32)
    # mask_valid_out = mask_valid_out[0, 0].numpy().astype(np.float32)
    # mask_active = mask_active[0].numpy().astype(np.float32)
    
    if depth_erp_gt is not None:
        depth_out_gt = F.grid_sample(depth_erp_gt.unsqueeze(0), erp_grid, mode='nearest', padding_mode='border', align_corners=True)
        depth_out_gt *= mask_active
        return img_out, depth_out, mask_valid_out, mask_active, depth_out_gt
    
    return img_out, depth_out, mask_valid_out, mask_active
    


def fisheye_mei_to_erp(fisheye_image, camera_params, output_size=(1400, 1400)):
    """
        Implement the inverse rendering, which is fast and leaves no holes
    """
    # Read intrinsics
    xi = camera_params['xi']
    k1 = camera_params['k1']
    k2 = camera_params['k2']
    p1 = camera_params['p1']
    p2 = camera_params['p2']
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']
            
    erp_height = output_size[0]
    erp_width = output_size[1]

    latitude = np.linspace(-np.pi / 2, np.pi / 2, erp_height)
    longitude = np.linspace(np.pi, 0, erp_width)
    # np.meshgrid defaut indexing is 'xy', which is different from 'ij' in torch
    longitude, latitude = np.meshgrid(longitude, latitude)
    longitude = longitude.astype(np.float32)
    latitude = latitude.astype(np.float32)

    x = np.cos(latitude) * np.cos(longitude)
    z = np.cos(latitude) * np.sin(longitude)
    y = np.sin(latitude)
    
    mask = z < 0
    x = np.where(mask, 1/np.sqrt(3), x)
    y = np.where(mask, 1/np.sqrt(3), y)
    z = np.where(mask, 1/np.sqrt(3), z)

    p_u = x / (z + xi)
    p_v = y / (z + xi)

    # apply distortion
    ro2 = p_u*p_u + p_v*p_v

    p_u *= 1 + k1*ro2 + k2*ro2*ro2
    p_v *= 1 + k1*ro2 + k2*ro2*ro2

    p_u += 2*p1*p_u*p_v + p2*(ro2 + 2*p_u*p_u)
    p_v += p1*(ro2 + 2*p_v*p_v) + 2*p2*p_u*p_v

    # apply projection
    p_u = fx*p_u + cx 
    p_v = fy*p_v + cy

    # Remap the fisheye image to ERP projection
    if fisheye_image.ndim == 2:
        erp_img = cv2.remap(fisheye_image, p_u, p_v, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        erp_img = cv2.remap(fisheye_image, p_u, p_v, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

    return erp_img

def erp_to_fisheye_mei(erp_img, camera_params, rotate):
    """
      This forward rendering can lead to some holes, but the inverse can be slow due to numerical optimization
    """

    width, height = camera_params['image_width'], camera_params['image_height']
    # erp_img =  np.roll(erp_img, 100, axis=0)

    xi = camera_params['xi']
    k1 = camera_params['k1']
    k2 = camera_params['k2']
    p1 = camera_params['p1']
    p2 = camera_params['p2']
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']

    erp_width, erp_height = erp_img.shape[1], erp_img.shape[0]

    latitude = np.linspace(-np.pi / 2, np.pi / 2, erp_height)
    longitude = np.linspace(np.pi, -np.pi, erp_width)
    
    longitude, latitude = np.meshgrid(longitude, latitude)
    longitude = longitude.astype(np.float32)
    latitude = latitude.astype(np.float32)

    x = np.cos(latitude) * np.cos(longitude)
    z = np.cos(latitude) * np.sin(longitude)
    y = np.sin(latitude)  

    xyz = np.stack((x,y,z),axis=2)

    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    [R1, _] = cv2.Rodrigues(x_axis * np.radians(rotate[0]))
    [R2, _] = cv2.Rodrigues(y_axis * np.radians(rotate[1]))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape([erp_height * erp_width, 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([erp_height , erp_width, 3])

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    # remove points with z < 0 without changing the shape of the arrays
    mask_z = z < 0
    erp_masked = np.where(mask_z[..., np.newaxis], 0, erp_img)

    x = np.where(mask_z, 1/np.sqrt(3), x)
    y = np.where(mask_z, 1/np.sqrt(3), y)
    z = np.where(mask_z, 1/np.sqrt(3), z)

    # convert to axuiliary coordinates
    p_u = x / (z + xi)
    p_v = y / (z + xi)

    # apply distortion
    ro2 = p_u*p_u + p_v*p_v

    p_u *= 1 + k1*ro2 + k2*ro2*ro2
    p_v *= 1 + k1*ro2 + k2*ro2*ro2

    p_u += 2*p1*p_u*p_v + p2*(ro2 + 2*p_u*p_u)
    p_v += p1*(ro2 + 2*p_v*p_v) + 2*p2*p_u*p_v

    # apply projection
    p_u = fx*p_u + cx 
    p_v = fy*p_v + cy

    mask_dimention = (p_u < 0) | (p_u >= width) | (p_v < 0) | (p_v >= height)
    erp_masked = np.where(mask_dimention[..., np.newaxis], 0, erp_masked)

    circle_mask = np.sqrt((p_u - cx)**2 + (p_v - cy)**2) > (min(height, width) / 2 - 0.025 * min(height, width))
    erp_masked = np.where(circle_mask[..., np.newaxis], 0, erp_masked)
    
    coordinates = np.stack([p_u, p_v],axis=-1)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    coordinate_points = coordinates.reshape((-1, 2))

    # also interpolate, just some part the projection can be too sparse compared to the inverse way
    interpolated_colors = griddata(coordinate_points, erp_masked.reshape((-1, 3)), grid_points, method='linear')
    interpolated_colors = interpolated_colors.reshape((height, width, 3))

    return interpolated_colors, erp_masked, np.logical_or(np.logical_or(mask_z, mask_dimention), circle_mask)


def fisheye_kb_to_erp(fisheye_image, camera_params, output_size=(1400, 1400), depth_map=None):
    """
        Implement the inverse rendering, which is fast and leaves no holes
    """
    # Read parameters
    k1 = camera_params['k1']
    k2 = camera_params['k2']
    k3 = camera_params['k3']
    k4 = camera_params['k4']
    fx = camera_params['fl_x']
    fy = camera_params['fl_y']
    cx = camera_params['cx']
    cy = camera_params['cy']
            
    erp_height = output_size[0]
    erp_width = output_size[1]

    # think of the coord frame as x-right, y-down, z-forward
    latitude = np.linspace(-np.pi / 2, np.pi / 2, erp_height)
    longitude = np.linspace(np.pi, 0, erp_width)
    # np.meshgrid defaut indexing is 'xy', which is different from 'ij' in torch
    longitude, latitude = np.meshgrid(longitude, latitude)
    longitude = longitude.astype(np.float32)
    latitude = latitude.astype(np.float32)

    x = np.cos(latitude) * np.cos(longitude)
    z = np.cos(latitude) * np.sin(longitude)
    y = np.sin(latitude)
    
    mask = z < 0
    x = np.where(mask, 1/np.sqrt(3), x)
    y = np.where(mask, 1/np.sqrt(3), y)
    z = np.where(mask, 1/np.sqrt(3), z)

    a = x / (z+1e-9)
    b = y / (z+1e-9)
    r = np.sqrt(a*a + b*b)
    theta = np.arctan(r)

    # apply distortion
    theta_d = theta * (1 + k1*theta*theta + k2*theta**4 + k3*theta**6 + k4*theta**8)
    
    x_d = theta_d * a / (r+1e-9)
    y_d = theta_d * b / (r+1e-9)
    
    u_d = fx * x_d + cx
    v_d = fy * y_d + cy

    # Remap the fisheye image to ERP projection
    erp_img = cv2.remap(fisheye_image, u_d, v_d, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    input_h, input_w = fisheye_image.shape[:2]
    active_mask = np.logical_and(np.logical_and(u_d > 0, u_d < input_w), np.logical_and(v_d > 0, v_d < input_h)) * 1.0
    if depth_map is not None:
        erp_depth = cv2.remap(depth_map, u_d, v_d, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return erp_img, erp_depth, active_mask
    return erp_img, active_mask