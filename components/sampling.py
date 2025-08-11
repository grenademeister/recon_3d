from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from common.utils import validate_tensor_dimensions
from common.wrapper import error_wrap


def ifft2c(img_k: Tensor) -> Tensor:
    img = torch.fft.ifftn(torch.fft.ifftshift(img_k, dim=(-2, -1)), dim=(-2, -1))
    return img


def fft2c(img: Tensor) -> Tensor:
    img_k = torch.fft.fftshift(torch.fft.fftn(img, dim=(-2, -1)), dim=(-2, -1))
    return img_k


def apply_fixed_mask(
    img: Tensor,
    acs_num: int,
    parallel_factor: int,
) -> tuple[
    Tensor,
    Tensor,
    None,
]:
    validate_tensor_dimensions(tensors=[img], expected_dim=4)

    acs_half = acs_num // 2
    img_k = fft2c(img)
    B, C, H, W = img.shape

    mask = torch.zeros([B, 1, H, W], dtype=torch.complex64)
    cen = mask.shape[3] // 2
    mask[:, :, :, cen - acs_half : cen + acs_half] = 1
    mask[:, :, :, ::parallel_factor] = 1
    mask = mask.to(img.device)

    output = ifft2c(img_k * mask)

    mask = mask.type(torch.float32)
    # mask is [B, 1, H, W] with float32 dtype

    return (
        output,
        mask.type(torch.float32),
        None,
    )


@lru_cache
def calc_k(
    encoding_line: int,
    acs_num: int,
    parallel_factor: int,
) -> int:
    mask = torch.zeros([encoding_line])

    acs_half = acs_num // 2

    cen = mask.shape[0] // 2
    mask[cen - acs_half : cen + acs_half] = 1
    mask[::parallel_factor] = 1

    return int(mask.sum().item())


def output_to_mask_v1(
    x: Tensor,
    k: int,
    fixed_thresh: Tensor | None,
    mask_fix: bool,
    temperature: int = 100,
) -> tuple[Tensor, Tensor]:
    validate_tensor_dimensions(tensors=[x], expected_dim=2)
    B, H = x.shape
    prob = x
    prob_mean = torch.mean(prob, dim=(1), keepdim=True)
    alpha = torch.ones_like(prob_mean) * k / H

    prob_rev1 = alpha / prob_mean * prob
    prob_rev2 = 1 - (1 - alpha) / (1 - prob_mean) * (1 - prob)

    prob_rev = prob_rev2 + ((prob_mean > alpha).float()) * (prob_rev1 - prob_rev2)

    if mask_fix:
        fixed_thresh = fixed_thresh[:, :H]
        thresh = fixed_thresh.repeat(B, 1)
    else:
        thresh = torch.rand_like(x).to(x.device)

    mask = torch.sigmoid(temperature * (prob_rev - thresh))
    return mask, prob_rev


@error_wrap
def apply_adaptive_mask(
    img: Tensor,
    sampling_output: Tensor,
    acs_num: int,
    parallel_factor: int,
    quantize: bool,
    fixed_thresh: Tensor | None,
    mask_fix: bool,
    variational_k: bool,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
]:
    validate_tensor_dimensions(tensors=[img], expected_dim=4)
    B, C, H, W = img.shape

    k = calc_k(
        encoding_line=H,
        acs_num=acs_num,
        parallel_factor=parallel_factor,
    )

    if variational_k:
        k = np.random.randint(51, max(52, k))

    mask, mask_prob = output_to_mask_v1(
        x=sampling_output,
        k=k,
        fixed_thresh=fixed_thresh,
        mask_fix=mask_fix,
    )

    mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, H, 1)
    mask_prob = mask_prob.unsqueeze(1).unsqueeze(1).repeat(1, 1, H, 1)

    if quantize:
        thresh, _ = torch.topk(mask, k, dim=3)
        mask = (mask >= thresh[:, :, :, -1].unsqueeze(-1).repeat(1, 1, 1, W)).float()

    mask = mask.type(torch.complex64)
    output = ifft2c(fft2c(img) * mask)

    return (
        output,
        mask.type(torch.float32),
        mask_prob,
    )
