import gc
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def patch_match(
        imgs: List[Image.Image],
        ft: torch.tensor,
        source_xy: Union[np.ndarray, List, tuple],
        patch_size=15,
        return_match_scores=False
) -> Union[tuple[List, np.ndarray, List[np.ndarray]], tuple[List, np.ndarray, List[np.ndarray], List[float]]]:
    """
    Perform patch matching between the source image and the target images.

    Args:
    - imgs (List[Image.Image]): The list of target images.
    - ft (torch.tensor): The feature tensor of the source image.
    - source_xy (Union[np.ndarray, List, tuple]): The source coordinates.
    - patch_size (int, optional): The patch size. Defaults to 15.
    - return_match_scores (bool, optional): Whether to return the match scores. Defaults to False.

    Returns:
    - best_match_yxs (List): The best match coordinates.
    - match_order (np.ndarray): The match order.
    - heatmaps (List[np.ndarray]): The heatmaps.
    - best_match_scores (List[float]): The best match scores.
    """
    num_imgs = len(imgs)
    K = patch_size
    half_patch_size = (patch_size - 1) // 2

    x, y = int(np.round(source_xy[0])), int(np.round(source_xy[1]))

    src_ft = ft[0].unsqueeze(0)
    src_patch = src_ft[0, :, y - half_patch_size:y + half_patch_size + 1,
                x - half_patch_size:x + half_patch_size + 1]  # 1, C, K, K

    src_rotated_patches = []

    for angle in range(0, 360, 90):
        rotated_patch = TF.rotate(src_patch, angle, expand=False)
        # TODO: make the rotated_patch same size
        src_rotated_patches.append(rotated_patch)

    del src_ft
    del src_patch
    gc.collect()
    torch.cuda.empty_cache()

    src_rotated_patches = [F.normalize(patch, dim=1) for patch in src_rotated_patches]  # 1, C, K, K

    best_match_scores = []
    best_match_yxs = []
    heatmaps = []

    gc.collect()
    torch.cuda.empty_cache()

    for i in range(1, num_imgs):
        trg_ft = F.normalize(ft[i], dim=1).unsqueeze(0)  # 1, C, H, W

        score_maps = []
        K = K
        _, C, H, W = trg_ft.shape
        for src_rotated_patch in src_rotated_patches:
            padding = (K - 1) // 2
            padded_trg_ft = F.pad(trg_ft, (padding, padding, padding, padding), mode='constant',
                                  value=0)
            unfolded = F.unfold(padded_trg_ft, kernel_size=K, stride=1)
            unfolded = unfolded.view(1, C, K, K, -1)
            similarity = (src_rotated_patch.unsqueeze(-1) * unfolded).sum(dim=(1, 2, 3)) / (K * K)
            similarity_map = similarity.view(1, H, W).cpu().numpy()
            score_maps.append(similarity_map)
            del unfolded
            del padded_trg_ft
            gc.collect()
            torch.cuda.empty_cache()
        score_map = np.concatenate(score_maps, axis=0).max(axis=0)  # HxW

        max_yx = np.unravel_index(score_map.argmax(), score_map.shape)

        best_match_scores.append(score_map.max())
        best_match_yxs.append(max_yx)

        heatmap = score_map
        heatmap = (heatmap - np.min(heatmap)) / (
                np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]

        heatmaps.append(heatmap)

        del score_map
        gc.collect()

    argsort = np.argsort(np.array(best_match_scores))
    match_order = np.argsort(argsort)
    if return_match_scores:
        return best_match_yxs, match_order, heatmaps, best_match_scores
    else:
        return best_match_yxs, match_order, heatmaps


def visualize_match(
        source_img: Image.Image,
        target_imgs: list[Image.Image],
        source_xy: tuple[int, int],
        best_match_yxs: List[Union[np.ndarray, tuple]],
        match_order: np.ndarray,
        heatmaps: List[np.ndarray] = None,
        num_rows: int = None,
        save_path: str = None,
        save_and_show: bool = False
):
    num_targets = len(target_imgs)
    if num_rows is None:
        num_rows = int(np.floor(np.sqrt(num_targets)))
    num_columns = int(np.ceil(num_targets / num_rows) + 1)
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
    axs = np.atleast_2d(axs)
    for i in range(num_rows):
        axs[i][0].imshow(source_img)
        axs[i][0].scatter(source_xy[0], source_xy[1], c='r', s=100)
        axs[i][0].set_title(f'({source_xy[0]}, {source_xy[1]})', fontsize=25)
        axs[i][0].axis('off')
    for i in range(num_targets):
        best_match_yx = best_match_yxs[i]
        row = i // (num_columns - 1)
        column = i % (num_columns - 1) + 1

        axs[row][column].imshow(target_imgs[i])
        axs[row][column].scatter(best_match_yx[1], best_match_yx[0], c='r', s=100)
        if heatmaps is not None:
            axs[row][column].imshow(heatmaps[i], alpha=0.5)
        axs[row][column].set_title(f'{match_order[i]}, ({best_match_yx[1]}, {best_match_yx[0]})', fontsize=25)
        axs[row][column].axis('off')

    if save_path is not None:
        plt.savefig(save_path)
        if save_and_show:
            plt.show()
    else:
        plt.show()
    plt.close()
