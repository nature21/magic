import os
from typing import List

import imageio
import numpy as np
from PIL import Image as Image
from sapien import core as sapien
from tqdm import tqdm


def get_rgba_img(camera: sapien.CameraEntity) -> np.ndarray:
    """Get RGBA image from the camera."""
    camera.take_picture()
    rgba_img = camera.get_color_rgba()
    rgba_img = (rgba_img * 255).clip(0, 255).astype("uint8")
    return rgba_img


def save_imgs2dir(imgs: List[np.ndarray], directory: str):
    """Save a list of image_scene to a directory."""
    os.makedirs(directory, exist_ok=True)
    for i, img in enumerate(imgs):
        img = Image.fromarray(img)
        img.save(f"{directory}/{i:04d}.png")


def imgs2mp4(imgs: List[np.ndarray], filename: str, fps: int = 60):
    """Save a list of image_scene to a mp4 file."""
    dir = os.path.dirname(filename)
    os.makedirs(dir, exist_ok=True)
    # uniformly sample 10 * fps frames
    if len(imgs) > 10 * fps:
        imgs = imgs[::len(imgs) // (10 * fps)]
    writer = imageio.get_writer(filename, fps=fps)
    for img in tqdm(imgs, desc='Writing video'):
        writer.append_data(img)
    writer.close()


def get_depth_img(camera: sapien.CameraEntity):
    return -camera.get_float_texture('Position')[..., 2]


def uvz2world(camera: sapien.CameraEntity, uvz: np.ndarray) -> np.ndarray:
    """
    uvz: 1D np array with format [u, v, Z],
    where (u, v) are the coordinates on *axis 1 and axis 0*,
    and Z is the depth.
    """
    # Extract the 3x3 intrinsic camera matrix and the 4x4 extrinsic camera matrix.
    intrinsic_matrix = camera.get_intrinsic_matrix()
    extrinsic_matrix = camera.get_extrinsic_matrix()

    u, v, Z = uvz

    intrinsics_inverse = np.linalg.inv(intrinsic_matrix)

    uv_homogeneous = np.array([u, v, 1])
    xyz_camera = np.dot(intrinsics_inverse, uv_homogeneous) * Z

    xyz_camera_homogeneous = np.append(xyz_camera, 1)
    xyz_world_homogeneous = np.dot(np.linalg.inv(extrinsic_matrix), xyz_camera_homogeneous)

    xyz_world = xyz_world_homogeneous[:3] / xyz_world_homogeneous[3]

    return xyz_world


def get_world_point_from_coordinate(camera: sapien.CameraEntity, coordinate: tuple[int, int]) -> np.ndarray:
    camera.take_picture()
    depth_img = get_depth_img(camera)
    return uvz2world(camera, np.array([*coordinate[::-1], depth_img[coordinate]]))


def depth_image_to_pcd(camera: sapien.CameraEntity, depth_image: np.ndarray, image_like: bool = False) -> np.ndarray:
    height, width = depth_image.shape
    intrinsic_matrix = camera.get_intrinsic_matrix()
    extrinsic_matrix = camera.get_extrinsic_matrix()
    # Create meshgrid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten u, v, and depth arrays for easier computation
    u = u.flatten()
    v = v.flatten()

    depth = depth_image.flatten()

    # Filter out zero depth values to avoid processing points without valid depth
    valid = depth > 0
    u = u[valid]
    v = v[valid]
    depth = depth[valid]

    # Transform to camera frame
    uv_homogeneous = np.vstack((u, v, np.ones_like(u)))
    intrinsics_inverse = np.linalg.inv(intrinsic_matrix)
    xyz_camera = np.dot(intrinsics_inverse, uv_homogeneous) * depth

    xyz_camera_homogeneous = np.vstack((xyz_camera, np.ones(xyz_camera.shape[1])))

    # Transform to world coordinates
    xyz_world_homogeneous = np.dot(np.linalg.inv(extrinsic_matrix), xyz_camera_homogeneous)
    xyz_world = xyz_world_homogeneous[:3, :] / xyz_world_homogeneous[3, :]

    pcd = xyz_world.T

    if image_like:
        if pcd.size != height * width * 3:
            raise ValueError('the depth image contains invalid points, cannot output image like')
        pcd = pcd.reshape(height, width, 3)
    return pcd


def get_points_in_world(camera: sapien.CameraEntity, image_like: bool = False) -> np.ndarray:
    """
    must be called after scene.update_render()!
    """
    position = camera.get_float_texture('Position')
    points_opengl = position[..., :3][position[..., 3] < 1]
    # np.dot(model_matrix, np.dot(np.diag([1, -1, -1, 1]), extrinsic_matrix)) is identity matrix
    model_matrix = camera.get_model_matrix()
    pcd = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    if image_like:
        if pcd.size != np.product(position.shape[:2]) * 3:
            raise ValueError('the depth image contains invalid points, cannot output image like')
        pcd = pcd.reshape(*position.shape[:2], 3)
    return pcd
