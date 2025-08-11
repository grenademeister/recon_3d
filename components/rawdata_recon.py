import torch
from model.longitudinal_recon import LongRecon
from torch import Tensor

from common.utils import validate_tensor_dimensions, validate_tensors
from components.registration import rotate_and_translate_with_scipy


def tensor_5d_2_complex(
    img: Tensor,
) -> Tensor:
    validate_tensor_dimensions([img], 5)  # [B, Z, H, W, C]
    if img.shape[-1] != 2:
        raise ValueError(f"Expected last dimension to be 2, got {img.shape[-1]} instead.")

    img_complex = img[..., 0] + 1j * img[..., 1]
    return img_complex.type(torch.complex64)


def tensor_complex_2_5d(
    img: Tensor,
) -> Tensor:
    validate_tensor_dimensions([img], 4)  # [B, Z, H, W]

    img_real = img.real.unsqueeze(-1)
    img_imag = img.imag.unsqueeze(-1)
    img_5d = torch.cat([img_real, img_imag], dim=-1)
    return img_5d.type(torch.float32)


def run_undersample_and_registration(
    long_recon: LongRecon,
    prior: Tensor,
    target: Tensor,
    slice_num: int,
    z_middle: int,
    using_registration: bool,
    B: int,
    Z: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor | None,
]:
    prior_pad = torch.nn.functional.pad(prior, (0, 0, 0, 0, z_middle, z_middle), mode="constant", value=0)
    target_pad = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, 0, z_middle, z_middle), mode="constant", value=0)

    # Under-sampling
    target_undersample = torch.zeros_like(target)  # [B, Z, H, W, C]
    if using_registration:
        target_sudo_reconstruction = torch.zeros_like(prior)  # [B, Z, H, W]
        prior_registration = torch.zeros_like(prior)
        reg_pred_tot = torch.zeros((B, Z, 3), device=prior.device)
    else:
        target_sudo_reconstruction = None
        prior_registration = None

    for i in range(Z):
        prior_slice = prior_pad[:, i : i + slice_num, :, :]
        target_slice = target_pad[:, i : i + slice_num, :, :, :]
        (
            target_undersample_slice,
            mask,
            mask_prob,
        ) = long_recon.undersample_img(
            target_slice,
            quantize=True,
            variational_k=False,
        )
        target_undersample[:, i, :, :, :] = target_undersample_slice[:, z_middle : z_middle + 1, :, :, :]

        # Collect registration prior params
        if using_registration:
            label_undersample_complex = tensor_5d_2_complex(target_undersample_slice[:, z_middle : z_middle + 1, :, :, :])
            sudo_recon_input = torch.abs(label_undersample_complex)
            target_sudorecon_slice = long_recon.sudo_recon_img(sudo_recon_input)
            target_sudo_reconstruction[:, i, :, :] = target_sudorecon_slice

            prior_reg, reg_pred = long_recon.registration_prior_to_target(
                prior=prior_slice,
                target=target_sudorecon_slice,
            )

            reg_pred_tot[:, i, :] = reg_pred

    # Registration
    if using_registration:
        reg_pred = reg_pred_tot.median(dim=1, keepdim=True)[0]
        for i in range(Z):
            prior_slice = prior_pad[:, i : i + slice_num, :, :]
            prior_reg = rotate_and_translate_with_scipy(
                img=prior_slice,
                angle=-reg_pred[0, 0, 0].cpu(),
                tx=-reg_pred[0, 0, 1].cpu(),
                ty=-reg_pred[0, 0, 2].cpu(),
            ).to(prior.device)

            prior_registration[:, i, :, :] = prior_reg[:, z_middle, :, :]

    return (
        target_undersample,
        mask,
        mask_prob,
        target_sudo_reconstruction,
        prior_registration,
    )


def run_recon_net(
    long_recon: LongRecon,
    target: Tensor,
    prior_cond_pad: Tensor,
    target_undersample_pad: Tensor,
    mask: Tensor,
    meta: Tensor,
    slice_num: int,
    z_middle: int,
    Z: int,
    batch_size: int,
) -> Tensor:
    target_reconstruction = torch.zeros_like(target)
    if target_undersample_pad.shape[0] != 1:
        raise ValueError("Batch size must be 1 for whole brain reconstruction.")

    # Batch acceleration
    for i in range(0, Z, batch_size):
        actual_batch = min(batch_size, Z - i)
        idx_list = [i + j for j in range(actual_batch)]

        # Build batch
        # [B_infer, slice_num, H, W]
        prior_batch = torch.stack([prior_cond_pad[:, j : j + slice_num, :, :].squeeze(0) for j in idx_list])
        # [B_infer, slice_num, H, W, C]
        target_us_batch = torch.stack([target_undersample_pad[:, j : j + slice_num, :, :, :].squeeze(0) for j in idx_list])

        label_batch = target_us_batch[:, z_middle : z_middle + 1, :, :, :]
        label_cond = label_batch.transpose(1, 4).squeeze(-1)  # [B_infer, C, H, W]

        # meta broadcasting
        meta_batch = meta.expand(actual_batch, -1)
        mask_batch = mask.expand(actual_batch, -1, -1, -1)  # [B_infer, 1, H, W]

        # Inference
        recon_batch = long_recon.flow_reverse(
            label_cond=label_cond,
            prior_cond=prior_batch,
            meta=meta_batch,
            mask=mask_batch,
        )  # [B_infer, C, H, W]

        # Insert into target_reconstruction
        for j, z_idx in enumerate(idx_list):
            target_reconstruction[:, z_idx : z_idx + 1, :, :, :] = recon_batch[j : j + 1, ...].unsqueeze(-1).transpose(1, 4)
    return target_reconstruction


def longitudinal_recon_wholebrain(
    long_recon: LongRecon,
    prior: Tensor,
    target: Tensor,
    meta: Tensor,
    batch_size: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor | None,
]:
    """
    Args:
        long_recon (LongRecon): Longitudinal reconstruction model.
        prior (Tensor): Prior image tensor of shape [B, Z, H, W].
        target (Tensor): Target image tensor of shape [B, Z, H, W, C].
        meta (Tensor): Meta information tensor of shape [B, A].
    """
    validate_tensors([prior, target])
    validate_tensor_dimensions([prior], 4)  # [B, Z, H, W]
    validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C]

    B, Z, _, _ = prior.shape
    if B != 1:
        raise ValueError("Batch size must be 1 for whole brain reconstruction.")

    slice_num = long_recon.input_depth
    z_middle = slice_num // 2

    using_registration = long_recon.using_registration

    # Under-sampling
    (
        target_undersample,
        mask,
        mask_prob,
        target_sudo_reconstruction,
        prior_registration,
    ) = run_undersample_and_registration(
        long_recon=long_recon,
        prior=prior,
        target=target,
        slice_num=slice_num,
        z_middle=z_middle,
        using_registration=using_registration,
        B=B,
        Z=Z,
    )

    prior_cond = prior_registration if using_registration else prior
    prior_cond_pad = torch.nn.functional.pad(prior_cond, (0, 0, 0, 0, z_middle, z_middle), mode="constant", value=0)
    target_undersample_pad = torch.nn.functional.pad(target_undersample, (0, 0, 0, 0, 0, 0, z_middle, z_middle), mode="constant", value=0)

    # Reconstruction
    target_reconstruction = run_recon_net(
        long_recon=long_recon,
        target=target,
        prior_cond_pad=prior_cond_pad,
        target_undersample_pad=target_undersample_pad,
        mask=mask,
        meta=meta,
        slice_num=slice_num,
        z_middle=z_middle,
        Z=Z,
        batch_size=batch_size,
    )

    return (
        target_reconstruction,
        target_undersample,
        mask,
        mask_prob,
        target_sudo_reconstruction,
        prior_registration,
    )
