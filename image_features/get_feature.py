import gc
try :
    from image_features.extractor_sd import process_features_and_mask
except : 
    pass

from typing import List

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from image_features.extractor_dino import ViTExtractor
from image_features.utils.utils_correspondence import resize


def torch_pca(feature: torch.Tensor, target_dim: int = 256) -> torch.Tensor:
    """
    Perform Principal Component Analysis (PCA) on the input feature tensor.

    Parameters:
    - feature (torch.Tensor): The input tensor with shape (N, D), where N is the number of samples
      and D is the feature dimension.
    - target_dim (int, optional): The target dimension for the output tensor. Defaults to 256.

    Returns:
    - torch.Tensor: The transformed tensor with shape (N, target_dim).
    """
    mean = torch.mean(feature, dim=0, keepdim=True)
    centered_features = feature - mean
    U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
    reduced_features = torch.matmul(centered_features, V[:, :target_dim])

    return reduced_features


def compute_dino_feature(
        source_img: Image.Image,
        target_imgs: List[Image.Image],
        model_size: str = 'base',
        use_dino_v2: bool = True,
        stride: int = None,
        edge_pad: bool = False,
        pca: bool = False,
        pca_dim: int = 256,
        reusable_extractor: ViTExtractor = None
) -> tuple[torch.Tensor, List[Image.Image], List[Image.Image]]:
    """
    return: (result, resized_imgs, downsampled_imgs), where result is a tensor of shape (N, pca_dim, num_patches, num_patches),
        resized_imgs is a list of PIL image_scene resized to the input size of the dino model, and downsampled_imgs is a list
        of PIL image_scene resized to the output size of the dino model.
    """
    img_size = 840 if use_dino_v2 else 244
    model_dict = {'small': 'dinov2_vits14',
                  'base': 'dinov2_vitb14',
                  'large': 'dinov2_vitl14',
                  'giant': 'dinov2_vitg14'}

    model_type = model_dict[model_size] if use_dino_v2 else 'dino_vits8'
    layer = 11 if use_dino_v2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if use_dino_v2 else 'key'
    if stride is None:
        stride = 14 if use_dino_v2 else 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if reusable_extractor is None:
        extractor = ViTExtractor(model_type, stride, device=device)
    else:
        extractor = reusable_extractor
    patch_size = extractor.model.patch_embed.patch_size[0] if use_dino_v2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

    result = []

    original_imgs = [source_img] + target_imgs
    resized_imgs = [resize(img, img_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]

    for img in tqdm(resized_imgs, desc='Extracting dino feature'):
        with torch.no_grad():
            img_batch = extractor.preprocess_pil(img)
            img_desc = extractor.extract_descriptors(
                img_batch.to(device), layer, facet)  # 1,1,num_patches*num_patches, feature_dim
            result.append(img_desc)

    result = torch.concat(result, dim=0)  # N, 1, num_patches*num_patches, feature_dim
    if pca:
        N, _, _, feature_dim = result.shape
        result = result.reshape(-1, feature_dim)
        result = torch_pca(result, pca_dim)
        result = result.reshape(N, 1, -1, pca_dim)

    result = result.permute(0, 1, 3, 2).reshape(result.shape[0], result.shape[-1], num_patches, num_patches)
    result = F.normalize(result, dim=1)

    gc.collect()
    torch.cuda.empty_cache()

    output_size = result.shape[-1]
    downsampled_imgs = [resize(img, output_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]

    return result, resized_imgs, downsampled_imgs


def compute_sd_dino_feature(
        source_img: Image.Image,
        target_imgs: List[Image.Image],
        sd_model,
        sd_aug,
        model_size: str = 'base',
        use_dino_v2: bool = True,
        stride: int = None,
        edge_pad: bool = False,
        pca: bool = False,
        pca_dim: int = 256,
        reusable_extractor: ViTExtractor = None
) -> tuple[torch.Tensor, List[Image.Image], List[Image.Image]]:
    """
    return: (result, resized_imgs, downsampled_imgs), where result is a tensor of shape (N, pca_dim, num_patches, num_patches),
        resized_imgs is a list of PIL image_scene resized to the input size of the dino model, and downsampled_imgs is a list
        of PIL image_scene resized to the output size of the dino model.
    """
    img_size = 840 if use_dino_v2 else 244
    real_size = 960
    model_dict = {'small': 'dinov2_vits14',
                  'base': 'dinov2_vitb14',
                  'large': 'dinov2_vitl14',
                  'giant': 'dinov2_vitg14'}

    model_type = model_dict[model_size] if use_dino_v2 else 'dino_vits8'
    layer = 11 if use_dino_v2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if use_dino_v2 else 'key'
    if stride is None:
        stride = 14 if use_dino_v2 else 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if reusable_extractor is None:
        extractor = ViTExtractor(model_type, stride, device=device)
    else:
        extractor = reusable_extractor
    patch_size = extractor.model.patch_embed.patch_size[0] if use_dino_v2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

    result = []

    original_imgs = [source_img] + target_imgs
    resized_imgs = [resize(img, img_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]
    sd_imgs = [resize(img, real_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]

    for img in tqdm(resized_imgs, desc='Extracting dino features'):
        with torch.no_grad():
            img_batch = extractor.preprocess_pil(img)
            img_desc = extractor.extract_descriptors(
                img_batch.to(device), layer, facet)  # 1,1,num_patches*num_patches, feature_dim
            result.append(img_desc)

    result = torch.concat(result, dim=0)  # N, 1, num_patches*num_patches, feature_dim

    sd_results = []
    for sd_img in tqdm(sd_imgs, desc='Extracting sd features'):
        img_desc_sd = process_features_and_mask(sd_model, sd_aug, sd_img, input_text=None,
                                                mask=False)  # 1, feature_dim, num_patches, num_patches
        sd_results.append(img_desc_sd)

    sd_results = torch.concat(sd_results, dim=0)  # N, feature_dim, num_patches, num_patches
    sd_results = sd_results.permute(0, 2, 3, 1).reshape(sd_results.shape[0], 1, -1, sd_results.shape[
        1])  # N, 1, num_patches*num_patches, feature_dim
    result = torch.cat((result, sd_results), dim=-1)

    if pca:
        N, _, _, feature_dim = result.shape
        result = result.reshape(-1, feature_dim)
        result = torch_pca(result, pca_dim)
        result = result.reshape(N, 1, -1, pca_dim)

    result = result.permute(0, 1, 3, 2).reshape(result.shape[0], result.shape[-1], num_patches, num_patches)
    result = F.normalize(result, dim=1)

    gc.collect()
    torch.cuda.empty_cache()

    output_size = result.shape[-1]
    downsampled_imgs = [resize(img, output_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]

    return result, resized_imgs, downsampled_imgs


def compute_dift_feature(
        source_img: Image.Image,
        target_imgs: List[Image.Image],
        dift_model,
        edge_pad: bool = False,
        pca: bool = False,
        pca_dim: int = 256,
) -> tuple[torch.Tensor, List[Image.Image], List[Image.Image]]:
    """
    return: (result, resized_imgs, downsampled_imgs), where result is a tensor of shape (N, pca_dim, num_patches, num_patches),
        resized_imgs is a list of PIL image_scene resized to the input size of the dino model, and downsampled_imgs is a list
        of PIL image_scene resized to the output size of the dino model.
    """
    img_size = 768
    emsembele_size = 4
    t = 261
    up_ft_index = 2
    downsampled_size = 96

    result = []

    original_imgs = [source_img] + target_imgs
    resized_imgs = [resize(img, img_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]

    for img in tqdm(resized_imgs, desc='Extracting dift features'):
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        ft, _ = dift_model.forward(
            img_tensor,
            ensemble_size=emsembele_size,
            t=t,
            up_ft_index=up_ft_index
        )  # 1, 640, 96, 96
        result.append(ft)

    result = torch.concat(result, dim=0)  # N, 640, 96, 96
    result = result.permute(0, 2, 3, 1).reshape(result.shape[0], 1, -1, result.shape[1])
    # N, 1, num_patches*num_patches, 640

    if pca:
        N, _, _, feature_dim = result.shape
        result = result.reshape(-1, feature_dim)
        result = torch_pca(result, pca_dim)
        result = result.reshape(N, 1, -1, pca_dim)

    result = result.permute(0, 1, 3, 2).reshape(result.shape[0], result.shape[-1], downsampled_size, downsampled_size)
    result = F.normalize(result, dim=1)

    gc.collect()
    torch.cuda.empty_cache()

    output_size = result.shape[-1]
    downsampled_imgs = [resize(img, output_size, resize=True, to_pil=True, edge=edge_pad) for img in original_imgs]

    return result, resized_imgs, downsampled_imgs

