import os
from typing import Callable, Union

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA

from magic.utils_2d import get_region, find_index_of_point, find_intersections_8_connectivity, get_nearest_point, \
    find_connected_points
from magic.visualization import visualize_local, visualize_global, visualize_alignment


def get_edge_points(img, canny_threshold: tuple[float, float] = (100, 200)) -> np.ndarray:
    """
    Extracts edge points from an image using the Canny edge detector.

    Returns:
        np.ndarray: An array of edge points with shape (n, 2), where each point is (x, y).
    """
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv.Canny(img, *canny_threshold)

    y_coords, x_coords = np.where(edges == 255)
    return np.column_stack([x_coords, y_coords])


def fit_quadratic(points: np.ndarray) -> float:
    """
    Fits a series of 2D points to the model y = ax^2 using Least Squares Regression.

    Parameters:
        points (np.ndarray): An array of points with shape (n, 2), where each point is (x, y).

    Returns:
        float: The coefficient 'a' in the model y = ax^2.
    """
    x = points[:, 0]
    y = points[:, 1]
    X = x ** 2
    X = X.reshape(-1, 1)
    a, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    return a[0]


def compute_pca(points: np.ndarray) -> tuple[np.ndarray, PCA]:
    """
    Applies PCA on a series of 2D points and returns the transformed points.

    Parameters:
        points (np.ndarray): An array of points with shape (n, 2).

    Returns:
        np.ndarray: Transformed points in the principal component space.
    """
    # Create a PCA object, number of components to keep is 2 since we have 2D data
    pca = PCA(n_components=2)
    pca.fit(points)
    transformed_points = pca.transform(points)
    return transformed_points, pca


def transform_pca(pca: PCA, pca_center: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transforms a point from the original space to the principal component space.

    Parameters:
        pca (PCA): The PCA object fitted on the original points.
        pca_center (np.ndarray): The center of the principal component space.
        point (np.ndarray): The point to transform.

    Returns:
        np.ndarray: The transformed point in the principal component space.
    """
    return pca.transform(point.reshape(-1, 2)) - pca_center


def remove_irrelevant_points(points: np.ndarray, view_point: Union[tuple, np.ndarray]) -> np.ndarray:
    """
    Removes points that are not visible from a specified view point.

    Parameters:
        points (np.ndarray): An array of points with shape (n, 2).
        view_point (Union[tuple, np.ndarray]): The point from which visibility is checked.

    Returns:
        np.ndarray: The points that are visible from the view point.
    """
    accepted_points = []
    for point in points:
        intersections, _ = find_intersections_8_connectivity(points, view_point, point)
        # if there exist an intersection with distance to the point larger than 10, remove the point
        if len(intersections) > 0:
            if np.any(np.linalg.norm(intersections - point, axis=1) > 10):
                continue
        accepted_points.append(point)

    return np.array(accepted_points)


def get_curvature_sign(
        img,
        center_of_contact: np.ndarray,
        center_of_curvature: np.ndarray,
        object_mask=None,
        black_background: bool = True,
        step_size: int = 5
) -> bool:
    """
    Determines the sign of the curvature based on the object in the image.

    Parameters:
        img (np.ndarray): The image containing the object.
        center_of_contact (np.ndarray): The center of the contact region.
        center_of_curvature (np.ndarray): The center of the curvature.
        object_mask (np.ndarray, optional): A binary mask of the object in the image.
        black_background (bool, optional): Whether the background of the image is black.
        step_size (int, optional): The step size to use when checking the object mask.

    Returns:
        bool: True if the contact point is convex, False if it is concave.
    """
    direction = (center_of_curvature - center_of_contact) / np.linalg.norm(center_of_curvature - center_of_contact)
    point_of_interest = center_of_contact + (direction * step_size).astype(int)
    if object_mask is not None:
        if object_mask[point_of_interest[1], point_of_interest[0]]:
            return True
        else:
            return False
    else:
        return img[point_of_interest[1], point_of_interest[0]] > 50 if black_background \
            else img[point_of_interest[1], point_of_interest[0]] < 200


def estimate_curvature(
        points: np.ndarray,
        center_of_contact: np.ndarray,
        radius_of_region: float,
        img: np.ndarray,
        object_mask: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, bool, Callable[[np.ndarray], np.ndarray]]:
    """
    Estimates the curvature of a 2D curve at a specific point on a given scale.

    Parameters:
        points (np.ndarray): An array of points with shape (n, 2).
        center_of_contact (np.ndarray): The center of the contact region.
        radius_of_region (float): The radius of the region around the point to consider.
        img (np.ndarray): The image containing the curve.
        object_mask (np.ndarray): A binary mask of the object in the image.

    Returns:
        tuple[np.ndarray, float, np.ndarray, bool, Callable[[np.ndarray], np.ndarray]]:
            - The center of the principal component space.
            - The radius of curvature.
            - The center of curvature.
            - The sign of the curvature.
            - A function to transform points to the principal component space.
    """
    tangent_region_points = get_region(center_of_contact, points, radius_of_region / 2)
    if len(tangent_region_points) <= 1:
        tangent_region_points = points
    _, pca = compute_pca(tangent_region_points)
    pca_region_points = pca.transform(points)
    pca_center = pca.transform(center_of_contact.reshape(1, 2))[0]
    calibrated_region_points = pca_region_points - pca_center
    a = fit_quadratic(calibrated_region_points)
    radius_of_curvature = 1 / (2 * a + 1e-6)
    center_of_curvature = pca.inverse_transform(np.array([0, radius_of_curvature]) + pca_center)
    curvature_sign = get_curvature_sign(img, center_of_contact, center_of_curvature, object_mask)

    def transform_pca_fn(point: np.ndarray) -> np.ndarray:
        return transform_pca(pca, pca_center, point)

    return pca_center, radius_of_curvature, center_of_curvature, curvature_sign, transform_pca_fn


def calculate_curvature_with_recompute(
        img,
        point: np.ndarray,
        radius_of_region: float,
        expected_curvature_sign: bool = None,
        object_mask=None,
        ablate_recompute: bool = False,
        viewpoint_step_size: int = 5,
        project_near: int = 5,
        project_far: int = 200
) -> tuple[bool, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Calculates the curvature of a 2D curve at a specific point on a given scale.

    Parameters:
        img (np.ndarray): The image containing the curve.
        point (np.ndarray): The point on the curve at which to calculate the curvature.
        radius_of_region (float): The radius of the region around the point to consider.
        expected_curvature_sign (bool, optional): The expected sign of the curvature.
        object_mask (np.ndarray, optional): A binary mask of the object in the image.
        ablate_recompute (np.ndarray, optional): remove the curvature recomputing for ablation study.
        viewpoint_step_size (int, optional): The step size to use when checking the object mask.
        project_near (int, optional): The near end when searching the contact point in the direction of the curvature.
        project_far (int, optional): The far end when searching the contact point in the direction of the curvature.

    Returns:
        tuple[bool, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], np.ndarray]:
            - The sign of the curvature.
            - The radius of curvature.
            - The center of curvature.
            - The center of the contact region.
            - The edge points of the curve.
            - The region points around the contact point.
            - A function to transform points to the principal component space.
    """
    edge_points = get_edge_points(img)
    center_of_contact = get_nearest_point(edge_points, point)
    region_points = get_region(center_of_contact, edge_points, radius_of_region)

    if ablate_recompute:
        pca_center, radius_of_curvature, center_of_curvature, curvature_sign, transform_pca_fn = estimate_curvature(
            region_points, center_of_contact, radius_of_region, img, object_mask)
        return (curvature_sign, radius_of_curvature, center_of_curvature, center_of_contact,
                edge_points, region_points, transform_pca_fn)

    index_of_contact_center = find_index_of_point(region_points, center_of_contact)
    connected_region_points = find_connected_points(region_points, index_of_contact_center)
    pca_center, radius_of_curvature, center_of_curvature, curvature_sign, transform_pca_fn = estimate_curvature(
        connected_region_points, center_of_contact, radius_of_region, img, object_mask)

    direction = (center_of_curvature - center_of_contact) / np.linalg.norm(center_of_curvature - center_of_contact)
    if expected_curvature_sign is not None and curvature_sign != expected_curvature_sign:
        sign = -1 if expected_curvature_sign else +1
        segment_start_point = center_of_contact + (direction * project_near * sign).astype(int)
        segment_end_point = center_of_contact + (direction * project_far * sign).astype(int)
        try:
            center_of_contact = get_nearest_point(
                find_intersections_8_connectivity(edge_points, segment_start_point, segment_end_point)[0],
                center_of_contact)
        except ValueError:
            pass  # if no intersection is found, keep the original center_of_contact
        region_points = get_region(center_of_contact, edge_points, radius_of_region)
        index_of_contact_center = find_index_of_point(region_points, center_of_contact)

        # visualize region_points
        # import matplotlib.pyplot as plt
        # plt.plot(region_points[:, 0], region_points[:, 1], 'ro')
        # plt.plot(center_of_contact[0], center_of_contact[1], 'bo')
        # plt.savefig('temp.png')
        # plt.close()
        #
        # raise NotImplementedError

        connected_region_points = find_connected_points(region_points, index_of_contact_center)
        view_point = center_of_contact + (direction * viewpoint_step_size).astype(int)
    else:
        view_point = (center_of_curvature + center_of_contact) / 2
    region_points = remove_irrelevant_points(connected_region_points, view_point)

    pca_center, radius_of_curvature, center_of_curvature, curvature_sign, transform_pca_fn = estimate_curvature(
        region_points, center_of_contact, radius_of_region, img, object_mask)

    return (curvature_sign, radius_of_curvature, center_of_curvature, center_of_contact,
            edge_points, region_points, transform_pca_fn)


def multi_scale_calculation(
        img,
        center: np.ndarray,
        radii_of_region: np.ndarray,
        expected_curvature_sign: bool = None,
        object_mask=None,
        use_recompute: bool = True,
        alpha: float = 3.5
) -> tuple[bool, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Calculates the curvature of a 2D curve at a specific point on multiple scales.

    Parameters:
        img (np.ndarray): The image containing the curve.
        center (np.ndarray): The point on the curve at which to calculate the curvature.
        radii_of_region (np.ndarray): An array of radii to consider.
        expected_curvature_sign (bool, optional): The expected sign of the curvature. True for convex, False for concave.
        object_mask (np.ndarray, optional): A binary mask of the object in the image.
        use_recompute (bool, optional): Whether to recompute the curvature if the expected sign does not match.
        alpha (float, optional): The ratio between the radii of the region and the curvature.

    Returns:
        tuple[bool, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
            - The sign of the curvature.
            - The radius of curvature.
            - The radius of the region.
            - The center of curvature.
            - The center of the contact region.
            - The edge points of the curve.
            - The region points around the contact point.
            - A function to transform points to the principal component space.
    """
    curvature_signs = []
    radii_of_curvature = []
    centers_of_curvature = []
    centers_of_contact = []
    edge_points_list = []
    region_points_list = []
    transform_pca_fns = []
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    for r in radii_of_region:
        (
            curvature_sign, radius_of_curvature, radius_center,
            contact_point, edge_points, region_points, transform_pca_fn
        ) = calculate_curvature_with_recompute(
            img, center, r, expected_curvature_sign, object_mask=object_mask, ablate_recompute=not use_recompute
        )
        curvature_signs.append(curvature_sign)
        radii_of_curvature.append(radius_of_curvature)
        centers_of_curvature.append(radius_center)
        centers_of_contact.append(contact_point)
        edge_points_list.append(edge_points)
        region_points_list.append(region_points)
        transform_pca_fns.append(transform_pca_fn)

    print(np.abs(radii_of_region / np.array(radii_of_curvature)))
    index = np.argmin(np.abs((np.abs(radii_of_region / np.array(radii_of_curvature))) - alpha))
    transform_pca_fn = transform_pca_fns[index]

    return (curvature_signs[index], radii_of_curvature[index], radii_of_region[index],
            centers_of_curvature[index], centers_of_contact[index],
            edge_points_list[index], region_points_list[index],
            transform_pca_fn)


def curv2d_alignment(
        img1,
        img2,
        center1: Union[tuple[float, float], np.ndarray],
        center2: Union[tuple[float, float], np.ndarray],
        save_dir: str = None,
        double_match: bool = False,
        collide_center_1: np.ndarray = None,
        collide_center_2: np.ndarray = None,
        human_view: bool = False,
        object_mask_1=None,
        object_mask_2=None,
        use_recompute: bool = True
) -> Union[tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
           tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Aligns two 2D curves by calculating the curvature at specific points on both curves.

    Parameters:
        img1 (np.ndarray): The image containing the first curve.
        img2 (np.ndarray): The image containing the second curve.
        center1 (Union[tuple[float, float], np.ndarray]): The point on the first curve at which to calculate the curvature.
        center2 (Union[tuple[float, float], np.ndarray]): The point on the second curve at which to calculate the curvature.
        save_dir (str, optional): The directory to save the visualization images.
        double_match (bool, optional): Whether to perform a double match. If True, the collision centers must be provided.
        collide_center_1 (np.ndarray, optional): The collision center on the first curve.
        collide_center_2 (np.ndarray, optional): The collision center on the second curve.
        human_view (bool, optional): Whether to use a human view for visualization.
        object_mask_1 (np.ndarray, optional): A binary mask of the object in the first image.
        object_mask_2 (np.ndarray, optional): A binary mask of the object in the second image.
        use_recompute (bool, optional): Whether to recompute with curvature sign correction and irrelevant point removal.
        local_search (bool, optional): Whether to perform a local search for the second curve. This is very slow.

    Returns:
        Union[tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            - The radius of curvature of the first curve.
            - The radius of curvature of the second curve.
            - The radius of the region of the first curve.
            - The radius of the region of the second curve.
            - The center of curvature of the first curve.
            - The center of curvature of the second curve.
            - The center of the contact region of the first curve.
            - The center of the contact region of the second curve.
            - The collision center on the second curve if double_match is True
    """
    radii_of_region = np.array(range(10, 110, 10))
    curvature_sign_1, radius_of_curvature_1, radius_of_region_1, center_of_curvature_1, center_of_contact_1, edge_points_1, region_points_1, transform_pca_fn_1 = multi_scale_calculation(
        img1, center1, radii_of_region, object_mask=object_mask_1, use_recompute=use_recompute)
    curvature_sign_2, radius_of_curvature_2, radius_of_region_2, center_of_curvature_2, center_of_contact_2, edge_points_2, region_points_2, transform_pca_fn_2 = multi_scale_calculation(
        img2, center2, radii_of_region, curvature_sign_1, object_mask=object_mask_2, use_recompute=use_recompute)

    visualize_local(transform_pca_fn_1(region_points_1), radius_of_region_1, radius_of_curvature_1,
                    save_path=os.path.join(save_dir, 'local_1.png') if save_dir is not None else None,
                    human_view=human_view)
    visualize_local(transform_pca_fn_2(region_points_2), radius_of_region_2, radius_of_curvature_2,
                    save_path=os.path.join(save_dir, 'local_2.png') if save_dir is not None else None,
                    human_view=human_view)

    visualize_global(edge_points_1, region_points_1, center_of_contact_1, center_of_curvature_1, radius_of_curvature_1,
                     save_path=os.path.join(save_dir, 'global_1.png') if save_dir is not None else None,
                     human_view=human_view)
    visualize_global(edge_points_2, region_points_2, center_of_contact_2, center_of_curvature_2, radius_of_curvature_2,
                     save_path=os.path.join(save_dir, 'global_2.png') if save_dir is not None else None,
                     human_view=human_view)

    radius_of_curvature_1 = np.abs(radius_of_curvature_1) * (1 if curvature_sign_1 else -1)
    radius_of_curvature_2 = np.abs(radius_of_curvature_2) * (1 if curvature_sign_2 else -1)

    vec1 = np.array([center_of_contact_1, center_of_curvature_1])
    vec2 = np.array([center_of_contact_2, center_of_curvature_2])

    visualize_alignment(vec1, vec2, edge_points_1, edge_points_2, region_points_1, region_points_2,
                        no_scaling=double_match,
                        save_path=os.path.join(save_dir, 'alignment.png') if save_dir is not None else None,
                        human_view=human_view)

    if double_match:
        assert collide_center_1 is not None and collide_center_2 is not None, \
            "collision centers must be provided for double match"
        visualize_alignment(vec1, vec2, edge_points_1, edge_points_2, region_points_1, region_points_2, no_scaling=True,
                            force_alignment=True, point_1=collide_center_1, point_2=collide_center_2,
                            save_path=os.path.join(save_dir, 'alignment_double.png') if save_dir is not None else None)
        return (radius_of_curvature_1, radius_of_curvature_2, radius_of_region_1, radius_of_region_2,
                center_of_curvature_1, center_of_curvature_2, center_of_contact_1, center_of_contact_2,
                collide_center_2)

    return (radius_of_curvature_1, radius_of_curvature_2, radius_of_region_1, radius_of_region_2,
            center_of_curvature_1, center_of_curvature_2, center_of_contact_1, center_of_contact_2)
