from typing import Union

import cv2 as cv
import networkx as nx
import numpy as np
from PIL import Image


def get_region(center: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    squared_distances = np.sum((points - center) ** 2, axis=1)
    return points[squared_distances <= radius ** 2]


def find_index_of_point(points: np.ndarray, point: np.ndarray) -> int:
    """
    Find the index of a specific point in an array of points.

    Parameters:
        points (np.ndarray): An array of points with shape (n, 2).
        point (np.ndarray): A single point with shape (2,).

    Returns:
        int: The index of the point in the points array, or -1 if not found.
    """
    # Check for matching points
    matches = np.all(points == point, axis=1)

    # Find the first index where matches is True
    indices = np.where(matches)[0]

    # Return the first matching index, or -1 if there are no matches
    return indices[0] if indices.size > 0 else -1


def calculate_transformation(vec1: np.ndarray, vec2: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Calculate the transformation matrix that aligns vec1 to vec2 including translation, rotation, and scaling.

    Parameters:
    - vec1, vec2: numpy arrays of shape (2,2), where each array contains the starting and ending points of the vectors.

    Returns:
    - Transformation matrix that aligns vec1 to vec2.
    """
    # Extract points
    start1, end1 = vec1[0], vec1[1]
    start2, end2 = vec2[0], vec2[1]

    # Translation to align starts
    translation = start2 - start1

    # Create direction vectors and normalize
    dir1 = end1 - start1
    dir2 = end2 - start2
    dir1_norm = np.linalg.norm(dir1)
    dir2_norm = np.linalg.norm(dir2)
    dir1_normalized = dir1 / dir1_norm
    dir2_normalized = dir2 / dir2_norm

    # Calculate rotation + scaling
    scale = dir2_norm / dir1_norm
    angle = np.arctan2(dir2_normalized[1], dir2_normalized[0]) - np.arctan2(dir1_normalized[1], dir1_normalized[0])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return scale, rotation_matrix


def apply_transformation(scale: float, rotation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a transformation matrix to a set of 2D points.

    Parameters:
    - transformation_matrix (np.ndarray): A 2x3 transformation matrix.
    - points (np.ndarray): An array of 2D points, shape (n, 2).

    Returns:
    - Transformed points as a numpy array, shape (n, 2).
    """
    # Apply the transformation matrix
    transformed = np.dot(points, rotation_matrix.T) * scale

    return transformed


def get_image(filename: str, rotation: float = None):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    if rotation is not None:
        img = Image.fromarray(img)
        img = img.rotate(rotation)
        img = np.array(img)
    return img


def get_line_pixels_opencv(xy1, xy2, width=768, height=768):
    """Get the pixels of a line segment using OpenCV's line drawing."""
    x1, y1 = xy1
    x2, y2 = xy2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Create an empty image
    image = np.zeros((height, width), dtype=np.uint8)

    # Draw the line
    cv.line(image, (y1, x1), (y2, x2), color=1, thickness=1)  # NOTE: use (y, x) instead of (x, y)!!!

    # Get the coordinates of the line pixels
    line_pixels = np.column_stack(np.where(image == 1))

    return line_pixels


def find_intersections_8_connectivity(
        edge_points: np.ndarray,
        xy1: Union[tuple, np.ndarray],
        xy2: Union[tuple, np.ndarray]
):
    """Find intersections of a line segment with an edge array using 8-connectivity."""
    line_pixels = get_line_pixels_opencv(xy1, xy2)

    neighbor_offsets = np.array([[dx, dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1]])

    line_pixels_buffer = (line_pixels[:, None, :] + neighbor_offsets).reshape(-1, 2)

    # Convert the 2D points to a 1D array of tuples
    # Define a structured data type for the points arrays
    dtype = [('x', np.int32), ('y', np.int32)]

    # Convert the arrays to this structured data type
    points_1_struct = np.array(list(map(tuple, line_pixels_buffer)), dtype=dtype)
    points_2_struct = np.array(list(map(tuple, edge_points)), dtype=dtype)

    # Find common elements using the structured arrays
    common_points_struct = np.intersect1d(points_1_struct, points_2_struct)

    # Convert the structured array back to a regular 2D NumPy array
    common_points = np.array(common_points_struct.tolist())

    return common_points, line_pixels


def get_coords_before_rotation(rotated_coords: np.ndarray, angle_deg: float, reflection: bool,
                               image_shape: np.ndarray) -> np.ndarray:
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)

    # Image dimensions
    width, height = image_shape

    # Calculate the center of the image
    cx, cy = width / 2, height / 2

    # Translate coordinates to center
    translated_coords = np.array([
        rotated_coords[0] - cx,
        rotated_coords[1] - cy
    ])

    # flip horizontally if reflection
    if reflection:
        translated_coords[0] = -translated_coords[0]

    # Rotation matrix for inverse rotation
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Apply the inverse rotation
    original_coords = np.dot(rotation_matrix, translated_coords)

    # Translate back from center
    original_coords += np.array([cx, cy])

    return original_coords


def find_nearest_point_with_mask(mask, point_xy, radius):
    # Define the bounds of the subarray
    x, y = point_xy
    x_min = max(0, x - radius)
    x_max = min(mask.shape[1], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(mask.shape[0], y + radius + 1)

    # Extract the subarray
    subarray = mask[y_min:y_max, x_min:x_max]

    # Find indices where the subarray has 1
    local_indices = np.where(subarray == 1)

    # Convert local indices to global indices
    points = np.array(local_indices).T + np.array([y_min, x_min])

    if len(points) > 0:
        distances = np.sqrt((points[:, 0] - y) ** 2 + (points[:, 1] - x) ** 2)
        min_index = np.argmin(distances)
        closest_point = points[min_index]
        return closest_point[::-1]
    else:
        raise ValueError("No points found within the radius.")


def get_nearest_point(points: np.ndarray, center1: np.ndarray, center2: np.ndarray = None) -> np.ndarray:
    """
    Finds the point in an array of points that is nearest to a (or two) specified point.

    Parameters:
        points (np.ndarray): An array of points with shape (n, 2).
        center1 (np.ndarray): The point to which the nearest point should be found.
        center2 (np.ndarray, optional): The second point to which the nearest point should be found.

    Returns:
        np.ndarray: The point in the array that is nearest to the specified point(s).
    """
    if center2 is not None:
        squared_distances = np.sum((points - center1) ** 2, axis=1) + np.sum((points - center2) ** 2, axis=1)
    else:
        squared_distances = np.sum((points - center1) ** 2, axis=1)
    nearest_point_index = np.argmin(squared_distances)
    return points[nearest_point_index]


def find_connected_points(points: np.ndarray, root_index: int, distance_threshold: float = 5) -> np.ndarray:
    """
    Finds all points connected to a specific point in an array of points.
    Two points are connected if the Euclidean distance between them is less
    than the specified distance threshold or there is a path connecting them.

    Parameters:
        points (np.ndarray): An array of points with shape (n, 2).
        root_index (int): The index of the point to which connected points should be found.
        distance_threshold (float, optional): The maximum Euclidean distance between points to be considered connected.

    Returns:
        np.ndarray: Array of points that are connected to the specified point.
    """
    # Calculate the pairwise squared Euclidean distances between points
    distances_squared = np.sum((points[:, np.newaxis] - points) ** 2, axis=2)
    threshold_squared = distance_threshold ** 2
    G = nx.Graph()
    # Add edges between nodes that are within the threshold distance
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if distances_squared[i, j] <= threshold_squared:
                G.add_edge(i, j)
    # Find all nodes in the same connected component as the node 'index'
    connected_nodes = list(nx.node_connected_component(G, root_index))
    connected_points = points[connected_nodes]
    return connected_points
