import os
import numpy as np
import re
import yaml
import sys

import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)


camera_params_02 = {
        "model_type": "MEI",
        "camera_name": "image_02",
        "image_width": 1400,
        "image_height": 1400,
        "mirror_parameters": {
            "xi": 2.2134047507854890e+00
        },
        "distortion_parameters": {
            "k1": 1.6798235660113681e-02,
            "k2": 1.6548773243373522e+00,
            "p1": 4.2223943394772046e-04,
            "p2": 4.2462134260997584e-04
        },
        "projection_parameters": {
            "gamma1": 1.3363220825849971e+03,
            "gamma2": 1.3357883350012958e+03,
            "u0": 7.1694323510126321e+02,
            "v0": 7.0576498308221585e+02
        }
    }

camera_params_03 = {
    "model_type": "MEI",
    "camera_name": "image_03",
    "image_width": 1400,
    "image_height": 1400,
    "mirror_parameters": {
        "xi": 2.5535139132482758e+00
    },
    "distortion_parameters": {
        "k1": 4.9370396274089505e-02,
        "k2": 4.5068455478645308e+00,
        "p1": 1.3477698472982495e-03,
        "p2": -7.0340482615055284e-04
    },
    "projection_parameters": {
        "gamma1": 1.4854388981875156e+03,
        "gamma2": 1.4849477411748708e+03,
        "u0": 6.9888316784030962e+02,
        "v0": 6.9814541887723055e+02
    }
}


if __name__=="__main__":
    import cv2
    import matplotlib.pyplot as plt

    output_path = 'splits/kitti360/'
    cam_id = 2
    # perspective
    if cam_id == 2:
        camera = camera_params_02
    elif cam_id == 3:
        camera = camera_params_03
    else:
        raise RuntimeError('Invalid Camera ID!')

    resize_factor = 0.5
    H = int(1400 * resize_factor)
    W = int(1400 * resize_factor)
    # [H, W]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # [H*W]
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5

    # [3, H*W]
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)

    grid = torch.from_numpy(pixels)

    map_dist = []
    z_dist = []
    
    k1 = camera['distortion_parameters']['k1']
    k2 = camera['distortion_parameters']['k2']
    p1 = camera['distortion_parameters']['p1']
    p2 = camera['distortion_parameters']['p2']
    gamma1 = camera['projection_parameters']['gamma1'] * resize_factor
    gamma2 = camera['projection_parameters']['gamma2'] * resize_factor
    u0 = camera['projection_parameters']['u0'] * resize_factor
    v0 = camera['projection_parameters']['v0'] * resize_factor
    mirror = camera['mirror_parameters']['xi']

    # ATTENTION, the range of ro2 can go beyond 1.0 for fisheye cameras, set it properly to reduce error
    for ro2 in torch.linspace(0.0, 5.0, 400000):
        ro2_after = np.sqrt(ro2) * (1 + k1*ro2 + k2*ro2*ro2)
        map_dist.append([(1 + k1*ro2 + k2*ro2*ro2), ro2_after])
    map_dist = np.array(map_dist)
    # print(map_dist)
    
    # TODO: z should have closed-form solution sqrt(1 - z^2) / (z + mirror) = sqrt(x^2 + y^2), don't need to search
    for z in torch.linspace(0.0, 1.0, 200000):
        z_after = np.sqrt(1 - z**2) / (z + mirror)
        z_dist.append([z, z_after])
    z_dist = np.array(z_dist)
    # print(z_dist)

    map_dist = torch.from_numpy(map_dist).cuda()
    z_dist = torch.from_numpy(z_dist).cuda()


    def chunk(grid):
        x = grid[0, :]
        y = grid[1, :]

        x = (x - u0) / gamma1
        y = (y - v0) / gamma2
        dist = torch.sqrt(x*x + y*y)
        indx = torch.abs(map_dist[:, 1:] - dist[None, :]).argmin(dim=0)
        x /= map_dist[indx, 0]
        y /= map_dist[indx, 0]
    
        z_after = torch.sqrt(x*x + y*y)
        indx = torch.abs(z_dist[:, 1:] - z_after[None, :]).argmin(dim=0)

        x *= (z_dist[indx, 0] +mirror)
        y *= (z_dist[indx, 0] +mirror)

        xy = torch.stack((x, y))
        return xy

    xys = []
    for i in range(H):
        xy = chunk(grid[:, i*W:(i+1)*W].cuda())
        xys.append(xy.permute(1, 0))
        if i % 10 == 0:
            print(i)
    xys = torch.cat(xys, dim=0)

    z = torch.sqrt(1. - torch.norm(xys, dim=1, p=2) ** 2)
    isnan = z.isnan()
    z[isnan] = 1.
    pcd = torch.cat((xys, z[:, None], isnan[:, None]), dim=1)
    print("saving grid")
    np.save(f'{output_path}/grid_fisheye_0{cam_id}.npy', pcd.detach().cpu().numpy().reshape(H, W, 4))

    """
        Treating each ray as a point on an unit sphere, apply forward distortion and project to compute the approximation error using the lookup table
    """
    def cam2image(pcd):
        p_u = pcd[:, 0] / (pcd[:, 2] + mirror)
        p_v = pcd[:, 1] / (pcd[:, 2] + mirror)
        z = pcd[:, 2]
        
        # apply distortion
        ro2 = p_u*p_u + p_v*p_v

        p_u *= 1 + k1*ro2 + k2*ro2*ro2
        p_v *= 1 + k1*ro2 + k2*ro2*ro2

        p_u += 2*p1*p_u*p_v + p2*(ro2 + 2*p_u*p_u)
        p_v += p1*(ro2 + 2*p_v*p_v) + 2*p2*p_u*p_v

        # apply projection
        p_u = gamma1*p_u + u0 
        p_v = gamma2*p_v + v0

        return p_u, p_v, z
    
    # import ipdb; ipdb.set_trace()
    # show error map
    x, y, d = cam2image(pcd[:, :3])

    error = (x - grid[0].cuda()) ** 2 + (y - grid[1].cuda()) ** 2

    error_map = error.reshape(H, W).detach().cpu().numpy()
    error_map = np.clip(error_map, 0, 30)
    plt.imshow(error_map)
    plt.show()

