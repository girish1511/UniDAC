if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import json
    from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
    import open3d as o3d
    from e3nn import o3
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image
    import tqdm

    level5data = LyftDataset(data_path='datasets//lyft-3d-object-detection', json_path='datasets//lyft-3d-object-detection/train_data', verbose=True)
    # level5data.list_categories()
    # level5data.list_attributes()
    # level5data.list_scenes()
    # print("number of samples", len(level5data.sample))
    # print("sample data", level5data.sample_data[0])
    # print("sample[0]", level5data.sample[0])
    scene = level5data.scene[0]
    print("scene", scene)

    print("Number of scenes", len(level5data.scene))

    my_sample_token = scene["first_sample_token"]
    my_sample = level5data.get('sample', my_sample_token)
    # print("my sample", my_sample)
    # level5data.list_sample(my_sample['token'])
    # print("data in my sample", my_sample['data'])
    for k, v in my_sample['data'].items():
        print(k)

    sensor_channel = 'CAM_FRONT'
    my_sample_data = level5data.get('sample_data', my_sample['data'][sensor_channel])
    # print("my_sample_data", my_sample_data)
    img_H = int(my_sample_data['height'])
    img_W = int(my_sample_data['width'])
    # level5data.render_sample_data(my_sample_data['token'])
    img_path = my_sample_data['filename'] # here we get the rgb
    # print("img_path", img_path)

    # Now intrinsics. Intrinsics are in the json file. Reference the calibrated_sensor_token in the sample_data.
    calib_file_path = 'datasets/lyft-3d-object-detection/train_data/calibrated_sensor.json'
    with open(calib_file_path, 'r') as f:
        calib_data = json.load(f)
        # print("calib_data", calib_data)
        token = my_sample_data['calibrated_sensor_token']
        for i, di in enumerate(calib_data):
            if di['token'] == token:
                print("i", i)
                cam_in = di['camera_intrinsic'] # here we get the intrinsics
                cam_in = np.array(cam_in)
                cam_in = cam_in.reshape((3, 3))
                cam_R = di['rotation']
                cam_t = di['translation']
                print("cam_in", cam_in)
                break

    # Now depth. Depth needs to be computed by projecting lidar points into the image.
    # First, we need to get the lidar points.
    sensor_channel = 'LIDAR_TOP'
    my_sample_data = level5data.get('sample_data', my_sample['data'][sensor_channel])
    # level5data.render_sample_data(my_sample_data['token'])
    lidar_token = my_sample_data['token']

    # Get the calibrated_sensor_token
    lidar_token = my_sample_data['calibrated_sensor_token']

    # Get the calibrated_sensor R and T
    for i, di in enumerate(calib_data):
        if di['token'] == lidar_token:
            lidar_R = di['rotation']
            lidar_t = di['translation']
            break

    # data_root = os.path.join(code_root, 'metric3d_data/lyft_data')
    data_root = 'datasets/lyft-3d-object-detection/'
    rgb_root = os.path.join(data_root, 'rgb_val')
    depth_root = os.path.join(data_root, 'depth_val')

    # make a directory for rgb and depth images if doesn't exist code_root/data_root/rgb
    os.makedirs(rgb_root, exist_ok=True)
    os.makedirs(depth_root, exist_ok=True)

    files = []

    count = 0
    d_max = 0
    for i in range(len(level5data.sample)):
        my_sample = level5data.sample[i]
        if 'LIDAR_TOP' not in my_sample['data'] and 'LIDAR_FRONT_LEFT' not in my_sample['data'] and 'LIDAR_FRONT_RIGHT' not in my_sample['data']:
            count += 1
    print("Number of samples where there is no Lidar sensor", count)

    my_sample = level5data.sample[500]
    print("my sample", my_sample)
    print("Number of sensor channels in my sample", len(my_sample['data']))
    for k in my_sample['data'].keys():
        print(k)
    # sensor_channel = 'CAM_FRONT_LEFT'
    depth_scale = 200

    n_samples = len(level5data.sample) # 22680
    # select last 20% samples for validation
    n_samples = int(0.2 * n_samples)
    print("Number of samples", n_samples)

    sensor_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_ZOOMED']
    cam_ins = []
    # loop with tqdm
    for l in range(len(sensor_channels)):
        sensor_channel = sensor_channels[l]
        print(sensor_channel)
        my_sample_data = level5data.get('sample_data', my_sample['data'][sensor_channel])
        calibrated_sensor_token = my_sample_data['calibrated_sensor_token']
        for i, di in enumerate(calib_data):
            if di['token'] == calibrated_sensor_token:
                cam_in = di['camera_intrinsic']
                cam_in = np.array(cam_in)
                cam_in = cam_in.reshape((3, 3))
                # convert cam_in in the form [fx, fy, cx, cy] list
                cam_in = [cam_in[0, 0], cam_in[1, 1], cam_in[0, 2], cam_in[1, 2]]
                cam_ins.append(cam_in)
                # print("cam_in", cam_in)
                break

        for n in tqdm.tqdm(range(n_samples)):
            # my_sample = level5data.sample[n]
            # index samples from the end for validation
            my_sample = level5data.sample[-n]
            l_explorer = LyftDatasetExplorer(level5data)
            points, z_vals = [], []
            cam_token = my_sample['data'][sensor_channel]
            try:
                lidar_token_1 = my_sample['data']['LIDAR_TOP']
                points_1, _, _ = l_explorer.map_pointcloud_to_image(lidar_token_1, cam_token)
                points_1 = points_1.T
                z_val_1 = points_1[:, 2].reshape(-1, 1)
                points_1 = points_1[:, :2]
                points.append(points_1)
                z_vals.append(z_val_1)
            except:
                pass
            try:
                lidar_token_2 = my_sample['data']['LIDAR_FRONT_RIGHT']
                points_2, _, _ = l_explorer.map_pointcloud_to_image(lidar_token_2, cam_token)
                points_2 = points_2.T
                z_val_2 = points_2[:, 2].reshape(-1, 1)
                points_2 = points_2[:, :2]
                points.append(points_2)
                z_vals.append(z_val_2)
            except:
                pass
            try:
                lidar_token_3 = my_sample['data']['LIDAR_FRONT_LEFT']
                points_3, _, _ = l_explorer.map_pointcloud_to_image(lidar_token_3, cam_token)
                points_3 = points_3.T
                z_val_3 = points_3[:, 2].reshape(-1, 1)
                points_3 = points_3[:, :2]
                points.append(points_3)
                z_vals.append(z_val_3)
            except:
                pass

            sample_data = level5data.get('sample_data', cam_token)
            rgb_path = sample_data['filename']
            rgb_path = osp.join(data_root, rgb_path)
            img_H = int(sample_data['height'])
            img_W = int(sample_data['width'])
            
            depth = np.zeros((img_H, img_W))
            count = np.zeros((img_H, img_W))
            # for i in range(len(points)): # Projecting points from all the lidars available in a sample onto the camera plane
            #     for j in range(points[i].shape[0]):
            #         x, y = points[i][j]
            #         x = int(np.round(x))
            #         y = int(np.round(y))
            #         if x >= 0 and x < img_W and y >= 0 and y < img_H:
            #             depth[y, x] += z_vals[i][j, 0]
            #             count[y, x] += 1
            for i in range(len(points)): # Projecting points from all the lidars available in a sample onto the camera plane
                valid_points = (points[i][:, 0] >= 0) & (points[i][:, 0] < img_W) & (points[i][:, 1] >= 0) & (points[i][:, 1] < img_H)
                depth[points[i][valid_points, 1].astype(int), points[i][valid_points, 0].astype(int)] += z_vals[i][valid_points, 0]
                count[points[i][valid_points, 1].astype(int), points[i][valid_points, 0].astype(int)] += 1
                
            count[count == 0] = 1
            depth = depth / count
            max_depth = np.max(depth)
            if max_depth > d_max:
                d_max = max_depth

            # Convert depth into 16 bit map
            depth = depth * depth_scale
            depth = depth.astype(np.uint16)

            # Save the depth image and the rgb image as well in a folder
            rgb_img = Image.open(rgb_path)
            rgb_img.save(rgb_root + f'/rgb_{n}_{l}.png')
            Image.fromarray(depth).save(depth_root + f'/depth_{n}_{l}.png')
            # print("Saved depth image", depth_root + f'/depth_{n}_{l}.png')

            rgb_path = f"lyft_data/rgb_val/rgb_{n}_{l}.png"
            depth_path = f"lyft_data/depth_val/depth_{n}_{l}.png"

            meta_data = {}
            meta_data['rgb'] = rgb_path
            meta_data['depth'] = depth_path
            meta_data['cam_in'] = cam_in
            # meta_data['depth_scale'] = depth_scale
            files.append(meta_data)
    files_dict = dict(files=files)
    
    with open('splits/lyft/val_annotations.json', 'w') as f:
        json.dump(files_dict, f)
    print('Saved annotations to', 'splits/lyft/val_annotations.json')
    print("d_max", d_max)