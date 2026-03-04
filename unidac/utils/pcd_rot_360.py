import numpy as np
import os
from PIL import Image
import open3d as o3d
import tqdm

# Specify the input folder containing PCD files
input_dir = "show_dirs/m3d/hm3d+taskonomy+hypersim_erp_m3d_new_erp/val_pcds"
output_base_dir = "show_dirs/m3d/video/frames_open3d_360"

# Ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Create an OffscreenRenderer
width, height = 800, 800  # Resolution of the frames
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

# Loop through each PCD file in the input directory
for pcd_file in tqdm.tqdm(os.listdir(input_dir)):
    if not pcd_file.endswith(".ply"):  # Process only .ply files
        continue
    
    # Load the point cloud
    point_cloud_path = os.path.join(input_dir, pcd_file)
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    
    # Flip the point cloud along the Y-axis to correct orientation
    point_cloud.points = o3d.utility.Vector3dVector(
        np.asarray(point_cloud.points) * np.array([1, -1, 1])
    )

    # Center the point cloud at the origin
    center = point_cloud.get_center()
    point_cloud.translate(-center)

    # Create an output directory for the current PCD file
    pcd_name = os.path.splitext(pcd_file)[0]
    output_dir = os.path.join(output_base_dir, pcd_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set the scene
    renderer.scene.set_background([1, 1, 1, 1])  # White background
    renderer.scene.add_geometry("point_cloud", point_cloud, o3d.visualization.rendering.MaterialRecord())

    # Configure the initial camera view
    bounds = point_cloud.get_axis_aligned_bounding_box()
    extent = bounds.get_extent().max()
    camera_distance = extent * 1.0  # Distance from the center
    up_vector = [0, 1, 0]  # Correct the up vector to align with the vertical Z-axis

    # Generate frames for rotation with continuous naming
    frame_counter = 0
    for angle in range(0, 360, 2):
        radian = np.deg2rad(angle)
        
        # Define camera position for rotation in the X-Z plane
        camera_position = [
            camera_distance * np.cos(radian),  # X-coordinate
            0,  # Fixed Y-coordinate
            camera_distance * np.sin(radian),  # Z-coordinate
        ]
        
        # Set the camera view to look at the origin
        renderer.scene.camera.look_at([0, 0, 0], camera_position, up_vector)

        # Render the scene to an image
        image = renderer.render_to_image()
        o3d.io.write_image(f"{output_dir}/frame_{frame_counter:03d}.png", image)
        frame_counter += 1  # Increment the frame counter

    # Combine frames into a GIF
    frames = [Image.open(f"{output_dir}/frame_{i:03d}.png") for i in range(frame_counter)]
    frames[0].save(os.path.join(output_dir, "point_cloud_rotation.gif"), save_all=True, append_images=frames[1:], duration=50, loop=0)

    # Clear geometry for the next iteration
    renderer.scene.clear_geometry()
