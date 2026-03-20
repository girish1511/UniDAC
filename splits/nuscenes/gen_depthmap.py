import os
import os.path as osp
import h5py
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from PIL import Image
import argparse



def gen_scene_depthmap(scene_name, occ_pts_file):
    scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    scene = nusc.get('scene',scene_token)
    sample_token = scene["first_sample_token"]

    camera_channel = "CAM_FRONT_RIGHT"#"CAM_BACK"#"CAM_BACK_RIGHT"#"CAM_BACK_LEFT"#"CAM_FRONT_LEFT"#"CAM_FRONT_RIGHT"#'CAM_FRONT'
    while sample_token:
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        camera_token = sample['data'][camera_channel]
        lidar = nusc.get('sample_data', lidar_token)
        cam = nusc.get('sample_data', camera_token)
        im_path = nusc.get_sample_data_path(camera_token)
        
        lidar_file = lidar['filename'].split('/')[-1].split('.')[0]
        cam_file = cam['filename'].split('/')[-1].split('.')[0]
        occ_mask = occ_pts_file[f"{lidar_file}@{cam_file}"][()]

        output_path = output_path.replace(".jpg",".png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pcl_path = os.path.join(nusc.dataroot, lidar['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        pc.points = pc.points[:, (occ_mask==0)]

        cs_record_lidar = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        cs_record_cam = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        poserecord_lidar = nusc.get('ego_pose', lidar['ego_pose_token'])
        poserecord_cam = nusc.get('ego_pose', cam['ego_pose_token'])

        lidar_to_veh_rot = Quaternion(cs_record_lidar['rotation']).rotation_matrix
        lidar_to_veh_trans = np.array(cs_record_lidar['translation'])
        lidar_to_veh_mat = geometric_transformation(lidar_to_veh_rot, lidar_to_veh_trans)

        cam_to_veh_rot = Quaternion(cs_record_cam['rotation']).rotation_matrix
        cam_to_veh_trans = np.array(cs_record_cam['translation'])
        cam_to_veh_mat = geometric_transformation(cam_to_veh_rot, cam_to_veh_trans)

        veh_to_glb_lidar_rot = Quaternion(poserecord_lidar['rotation']).rotation_matrix
        veh_to_glb_lidar_trans = np.array(poserecord_lidar['translation'])
        veh_to_glb_lidar_mat = geometric_transformation(veh_to_glb_lidar_rot, veh_to_glb_lidar_trans)

        veh_to_glb_cam_rot = Quaternion(poserecord_cam['rotation']).rotation_matrix
        veh_to_glb_cam_trans = np.array(poserecord_cam['translation'])
        veh_to_glb_cam_mat = geometric_transformation(veh_to_glb_cam_rot, veh_to_glb_cam_trans)
        relative_pose = np.linalg.inv(cam_to_veh_mat) @ np.linalg.inv(veh_to_glb_cam_mat) @ veh_to_glb_lidar_mat @ lidar_to_veh_mat

        pc.transform(relative_pose)
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        depths = pc.points[2, :]

        min_dist = 1.0
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]

        depthMap = get_int_depth_map(im.size[::-1], points.T, coloring)
        depthMap_im = Image.fromarray((depthMap*256).astype(np.uint16))

        depthMap_im.save(output_path)

        sample_token = sample["next"]
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate depth maps from nuScenes dataset')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    args = parser.parse_args()

    nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_dir, verbose=True)
    scene_splits = create_splits_scenes()


    occ_pts_filename = osp.join(args.data_dir, "nuscenes.h5")
    occ_pts_file = h5py.File(occ_pts_filename, 'r')

    for scene_name in scene_splits['val']:
        get_scene_depthmap(scene_name, occ_pts_file)