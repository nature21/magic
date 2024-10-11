import os

import numpy as np
from PIL import Image

from image_features.get_feature import compute_dino_feature

try:
    from image_features.get_feature import compute_sd_dino_feature
    from image_features.get_feature import compute_dift_feature
    from image_features.extractor_sd import load_model
    from image_features.extractor_dift import SDFeaturizer
except:
    pass

from image_features.match import patch_match, visualize_match
from magic.curvature import curv2d_alignment
from magic.utils_2d import get_coords_before_rotation


def match(
        source_img: Image.Image,
        target_img: Image.Image,
        source_center: tuple,
        grasp_center: tuple = None,
        model_size: str = 'base', use_dino_v2: bool = True,
        pca: bool = True, pca_dim: int = 256,
        parameter_save_dir: str = None,
        save_dir: str = 'results/temp',
        top_k: int = 3,
        patch_size=13,
        num_rotation=12,
        use_reflection=False,
        source_object_mask=None,
        target_object_mask=None,
        rotate_fill_color=(0, 0, 0),
        use_recompute=True,
        save_and_show=False,
        only_compute_dino_feature=False,
        sd_dino=False,
        dift=False,
):
    """
    Match the source image with the target image with rotation (and reflection) using DINO features and curvature.
    """
    os.makedirs(save_dir, exist_ok=True)
    angle = 360 // num_rotation

    target_imgs = [target_img.rotate(angle * i, fillcolor=rotate_fill_color) for i in range(num_rotation)]
    if use_reflection:
        reflected_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in target_imgs]
        target_imgs = target_imgs + reflected_imgs

    original_imgs = [source_img] + target_imgs

    if sd_dino and dift:
        raise ValueError('Cannot use both sd_dino and dift at the same time')

    if not sd_dino and not dift:
        result, resized_imgs, downsampled_imgs = compute_dino_feature(source_img, target_imgs, model_size=model_size,
                                                                      use_dino_v2=use_dino_v2, pca=pca, pca_dim=pca_dim)
    elif sd_dino:
        sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=960, num_timesteps=100)
        result, resized_imgs, downsampled_imgs = compute_sd_dino_feature(source_img, target_imgs, sd_model, sd_aug,
                                                                         model_size=model_size, use_dino_v2=use_dino_v2,
                                                                         pca=pca, pca_dim=pca_dim)
    else:
        dift_model = SDFeaturizer()
        result, resized_imgs, downsampled_imgs = compute_dift_feature(source_img, target_imgs, dift_model, pca=pca,
                                                                      pca_dim=pca_dim)

    if only_compute_dino_feature:
        return result, resized_imgs, downsampled_imgs

    original_size = original_imgs[0].size[0]
    downsampled_size = downsampled_imgs[0].size[0]
    ratio = original_size / downsampled_size
    source_xy = (int(source_center[0] / ratio), int(source_center[1] / ratio))
    best_match_yxs, match_order, heatmaps, match_scores = patch_match(resized_imgs, result, source_xy,
                                                                      patch_size=patch_size, return_match_scores=True)
    target_centers = np.round((np.array(best_match_yxs) + 0.5) * ratio).astype(int)
    visualize_match(original_imgs[0], original_imgs[1:], source_center, target_centers, match_order, None,
                    save_path=f'{save_dir}/contact_match.png', save_and_show=save_and_show)

    if grasp_center is not None:
        grasp_xy = (int(grasp_center[0] / ratio), int(grasp_center[1] / ratio))
        grasp_best_match_yxs, grasp_match_order, _ = patch_match(resized_imgs, result, grasp_xy, patch_size=13)
        grasp_target_centers = np.round((np.array(grasp_best_match_yxs) + 0.5) * ratio).astype(int)
        visualize_match(original_imgs[0], original_imgs[1:], grasp_center, grasp_target_centers, grasp_match_order,
                        None, save_path=f'{save_dir}/grasp_match.png', save_and_show=save_and_show)

    top_k_index = np.argsort(-match_order)[:top_k]
    alignment_results = []
    for i, best_rotation_index in enumerate(top_k_index):
        sub_dir = os.path.join(save_dir, f'{i}')
        os.makedirs(sub_dir, exist_ok=True)
        best_rotation = best_rotation_index * angle
        reflection = best_rotation_index >= num_rotation

        best_yx = best_match_yxs[best_rotation_index]
        target_center = np.round((np.array(best_yx) + 0.5) * ratio).astype(int)[::-1]

        if grasp_center is not None:
            best_grasp_yx = grasp_best_match_yxs[best_rotation_index]
            best_target_grasp_center = np.round((np.array(best_grasp_yx) + 0.5) * ratio).astype(int)[::-1]
            center_of_grasp_2 = get_coords_before_rotation(
                best_target_grasp_center,
                best_rotation,
                reflection,
                target_img.size
            )
        else:
            center_of_grasp_2 = None

        (radius_of_curvature_1, radius_of_curvature_2, radius_of_region_1, radius_of_region_2,
         center_of_curvature_1, center_of_curvature_2, center_of_contact_1, center_of_contact_2) \
            = curv2d_alignment(
            np.asarray(source_img),
            np.asarray(target_imgs[best_rotation_index]),
            source_center,
            target_center,
            save_dir=sub_dir,
            object_mask_1=np.asarray(source_object_mask) if source_object_mask is not None else None,
            object_mask_2=np.asarray(
                target_object_mask.rotate(best_rotation)) if target_object_mask is not None else None,
            use_recompute=use_recompute)

        # rotate the points of the target image to the orignal image
        center_of_contact_2 = get_coords_before_rotation(
            center_of_contact_2, best_rotation, reflection, target_img.size
        )
        center_of_curvature_2 = get_coords_before_rotation(
            center_of_curvature_2, best_rotation, reflection, target_img.size
        )

        alignment_result = {
            'radius_of_curvature_1': radius_of_curvature_1,
            'radius_of_curvature_2': radius_of_curvature_2,
            'center_of_contact_1': center_of_contact_1,
            'center_of_contact_2': center_of_contact_2,
            'center_of_curvature_1': center_of_curvature_1,
            'center_of_curvature_2': center_of_curvature_2,
            'center_of_grasp_2': center_of_grasp_2
        }
        alignment_results.append(alignment_result)

    # save to npy
    if parameter_save_dir is not None:
        os.makedirs(parameter_save_dir, exist_ok=True)
        np.save(os.path.join(parameter_save_dir, 'alignment_results.npy'), np.array(alignment_results))

    return alignment_results


def double_match(source_img, target_img, support_center, collide_center, grasp_center=None,
                 model_size='base', use_dino_v2=True, pca=True, pca_dim=256,
                 parameter_save_dir=None, save_dir='results/temp', top_k=3,
                 patch_size=13,
                 num_rotation=12,
                 use_reflection=False,
                 rotate_fill_color=(0, 0, 0),
                 use_recompute=True
                 ):
    """
    Under hard constraint encoded by collide_center,
    match the source image with the target image with rotation (and reflection) using DINO features and curvature.
    """
    os.makedirs(save_dir, exist_ok=True)
    angle = 360 // num_rotation

    target_imgs = [target_img.rotate(angle * i, fillcolor=rotate_fill_color) for i in range(num_rotation)]
    if use_reflection:
        reflected_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in target_imgs]
        target_imgs = target_imgs + reflected_imgs

    original_imgs = [source_img] + target_imgs

    result, resized_imgs, downsampled_imgs = compute_dino_feature(source_img, target_imgs, model_size=model_size,
                                                                  use_dino_v2=use_dino_v2, pca=pca, pca_dim=pca_dim)

    original_size = original_imgs[0].size[0]
    downsampled_size = downsampled_imgs[0].size[0]
    ratio = original_size / downsampled_size

    support_xy = (int(support_center[0] / ratio), int(support_center[1] / ratio))
    support_best_match_yxs, support_match_order, _ = patch_match(resized_imgs, result, support_xy,
                                                                 patch_size=patch_size)
    support_target_centers = np.round((np.array(support_best_match_yxs) + 0.5) * ratio).astype(int)
    visualize_match(original_imgs[0], original_imgs[1:], support_center, support_target_centers, support_match_order,
                    None, save_path=f'{save_dir}/support_match.png')

    collide_xy = (int(collide_center[0] / ratio), int(collide_center[1] / ratio))
    collide_best_match_yxs, collide_match_order, _ = patch_match(resized_imgs, result, collide_xy, patch_size=13)
    collide_target_centers = np.round((np.array(collide_best_match_yxs) + 0.5) * ratio).astype(int)
    visualize_match(original_imgs[0], original_imgs[1:], collide_center, collide_target_centers, collide_match_order,
                    None, save_path=f'{save_dir}/collide_match.png')

    if grasp_center is not None:
        grasp_xy = (int(grasp_center[0] / ratio), int(grasp_center[1] / ratio))
        grasp_best_match_yxs, grasp_match_order, _ = patch_match(resized_imgs, result, grasp_xy, patch_size=13)
        grasp_target_centers = np.round((np.array(grasp_best_match_yxs) + 0.5) * ratio).astype(int)
        visualize_match(original_imgs[0], original_imgs[1:], grasp_center, grasp_target_centers, grasp_match_order,
                        None, save_path=f'{save_dir}/grasp_match.png')

    top_k_index = np.argsort(-support_match_order - collide_match_order)[:top_k]  # TODO: figure this out
    alignment_results = []
    for i, best_rotation_index in enumerate(top_k_index):
        sub_dir = os.path.join(save_dir, f'{i}')
        os.makedirs(sub_dir, exist_ok=True)
        best_rotation = best_rotation_index * angle
        reflection = (best_rotation_index >= num_rotation)

        best_target_support_yx = support_best_match_yxs[best_rotation_index]
        best_target_support_center = np.round((np.array(best_target_support_yx) + 0.5) * ratio).astype(int)[::-1]

        best_target_collide_yx = collide_best_match_yxs[best_rotation_index]
        best_target_collide_center = np.round((np.array(best_target_collide_yx) + 0.5) * ratio).astype(int)[::-1]

        if grasp_center is not None:
            best_target_grasp_yx = grasp_best_match_yxs[best_rotation_index]
            best_target_grasp_center = np.round((np.array(best_target_grasp_yx) + 0.5) * ratio).astype(int)[::-1]

        (radius_of_curvature_1, radius_of_curvature_2, radius_of_region_1, radius_of_region_2,
         center_of_curvature_1, center_of_curvature_2, center_of_contact_1, center_of_contact_2, center_of_collide_2) \
            = curv2d_alignment(
            np.asarray(source_img),
            np.asarray(target_imgs[best_rotation_index]),
            support_center,
            best_target_support_center,
            save_dir=sub_dir,
            double_match=True,
            collide_center_1=collide_center,
            collide_center_2=best_target_collide_center,
            use_recompute=use_recompute
        )

        # rotate the points of the target image to the orignal image
        center_of_contact_2 = get_coords_before_rotation(center_of_contact_2, best_rotation, reflection,
                                                         target_img.size)
        center_of_curvature_2 = get_coords_before_rotation(center_of_curvature_2, best_rotation, reflection,
                                                           target_img.size)
        center_of_collide_2 = get_coords_before_rotation(center_of_collide_2, best_rotation, reflection,
                                                         target_img.size)
        if grasp_center is not None:
            center_of_grasp_2 = get_coords_before_rotation(best_target_grasp_center, best_rotation, reflection,
                                                           target_img.size)
        else:
            center_of_grasp_2 = None

        alignment_result = {
            'radius_of_curvature_1': radius_of_curvature_1,
            'radius_of_curvature_2': radius_of_curvature_2,
            'center_of_contact_1': center_of_contact_1,
            'center_of_contact_2': center_of_contact_2,
            'center_of_curvature_1': center_of_curvature_1,
            'center_of_curvature_2': center_of_curvature_2,
            'center_of_collide_2': center_of_collide_2,
            'center_of_grasp_2': center_of_grasp_2
        }
        alignment_results.append(alignment_result)

    # save to npy
    if parameter_save_dir is not None:
        os.makedirs(parameter_save_dir, exist_ok=True)
        np.save(os.path.join(parameter_save_dir, 'alignment_results.npy'), np.array(alignment_results))

    return alignment_results
