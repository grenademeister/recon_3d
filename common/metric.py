import torch
from torch import nn
from torch.nn import functional as F

IMG_DIM: int = 4


class SSIMcal(nn.Module):
    def __init__(self, win_size: int = 11, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        # single-channel averaging window
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / (win_size ** 2))
        npix = win_size ** 2
        self.cov_norm = npix / (npix - 1)

    def forward(
        self,
        img: torch.Tensor,        # (B, Z, H, W)
        ref: torch.Tensor,        # (B, Z, H, W)
        data_range: torch.Tensor, # (B,) or (B, 1, 1, 1)
    ) -> torch.Tensor:
        if img.dim() != 4 or ref.dim() != 4:
            raise ValueError("img and ref must be 4D tensors: (B, Z, H, W).")
        if img.shape != ref.shape:
            raise ValueError("img and ref must have the same shape.")

        B, Z, H, W = img.shape
        dev = img.device
        w = self.w.to(dev)

        # Flatten Z into batch to run depthwise conv2d with a single in-channel
        img_f = img.reshape(B * Z, 1, H, W)
        ref_f = ref.reshape(B * Z, 1, H, W)

        # Normalize data_range shape to (B*Z, 1, 1, 1)
        if data_range.dim() == 1:
            # (B,) -> (B, Z, 1, 1) -> (B*Z, 1, 1, 1)
            dr = data_range.view(B, 1, 1, 1).expand(B, Z, 1, 1).reshape(B * Z, 1, 1, 1)
        elif data_range.shape == (B, 1, 1, 1):
            dr = data_range.expand(B, Z, 1, 1).reshape(B * Z, 1, 1, 1)
        else:
            raise ValueError("data_range must be (B,) or (B,1,1,1).")
        C1 = (self.k1 * dr) ** 2
        C2 = (self.k2 * dr) ** 2

        ux  = F.conv2d(img_f, w)
        uy  = F.conv2d(ref_f, w)
        uxx = F.conv2d(img_f * img_f, w)
        uyy = F.conv2d(ref_f * ref_f, w)
        uxy = F.conv2d(img_f * ref_f, w)

        vx  = self.cov_norm * (uxx - ux * ux)
        vy  = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)

        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux ** 2 + uy ** 2 + C1
        B2 = vx + vy + C2

        S = (A1 * A2) / (B1 * B2)  # (B*Z, 1, H', W')
        # mean over spatial, then average across Z to return (B,1,1,1)
        ssim_spatial = S.mean(dim=(2, 3), keepdim=True)          # (B*Z,1,1,1)
        ssim_bz = ssim_spatial.view(B, Z, 1, 1, 1).mean(1, keepdim=False)  # (B,1,1,1)
        return ssim_bz


ssim_cal = SSIMcal()


def _maybe_complex_to_mag(x: torch.Tensor) -> torch.Tensor:
    # If Z==2, treat as 2-channel real/imag and convert to magnitude
    if x.shape[1] == 2:
        return torch.sqrt(x[:, :1, ...] ** 2 + x[:, 1:, ...] ** 2)
    return x


def calculate_ssim(
    img: torch.Tensor,  # (B, Z, H, W)
    ref: torch.Tensor,  # (B, Z, H, W)
    mask: torch.Tensor | None = None,  # optionally (B, 1 or Z, H, W)
) -> torch.Tensor:
    if not (img.dim() == IMG_DIM and ref.dim() == IMG_DIM):
        raise ValueError("All tensors must be 4D (B, Z, H, W).")
    if img.shape != ref.shape:
        raise ValueError("img and ref must have the same shape.")

    img = _maybe_complex_to_mag(img)
    ref = _maybe_complex_to_mag(ref)

    if mask is not None:
        if mask.dim() != IMG_DIM:
            raise ValueError("Mask must be 4D.")
        mask = _maybe_complex_to_mag(mask)
        # broadcast works for mask with channel 1 or Z
        img_mask = img * mask
        ref_mask = ref * mask
    else:
        img_mask, ref_mask = img, ref

    # Use unit data_range per batch; SSIMcal will expand across Z
    ones = torch.ones(img.shape[0], device=img.device)
    return ssim_cal(img_mask, ref_mask, ones)  # (B,1,1,1)


def calculate_psnr(
    img: torch.Tensor,  # (B, Z, H, W)
    ref: torch.Tensor,  # (B, Z, H, W)
    mask: torch.Tensor | None = None,  # optionally (B, 1 or Z, H, W)
) -> torch.Tensor:
    if not (img.dim() == IMG_DIM and ref.dim() == IMG_DIM):
        raise ValueError("All tensors must be 4D.")
    if img.shape != ref.shape:
        raise ValueError("img and ref must have the same shape.")

    img = _maybe_complex_to_mag(img)
    ref = _maybe_complex_to_mag(ref)

    if mask is not None:
        mask = _maybe_complex_to_mag(mask)
        img_mask = img * mask
        ref_mask = ref * mask
        mse = torch.sum((img_mask - ref_mask) ** 2, dim=(1, 2, 3), keepdim=True) / (
            torch.sum(mask, dim=(1, 2, 3), keepdim=True) + 1e-12
        )
    else:
        # mean over Z,H,W per batch
        mse = torch.mean((img - ref) ** 2, dim=(1, 2, 3), keepdim=True)

    img_max = torch.amax(ref, dim=(1, 2, 3), keepdim=True)
    psnr = 10 * torch.log10((img_max ** 2) / (mse + 1e-12))
    return psnr

if __name__ == "__main__":
    torch.manual_seed(0)

    # Example: batch=2, depth=5, height=32, width=32
    B, Z, H, W = 2, 5, 32, 32
    img = torch.rand(B, Z, H, W)
    ref = torch.rand(B, Z, H, W)

    # Optional mask
    mask = torch.ones(B, 1, H, W)
    mask[:, :, 10:20, 10:20] = 0.0  # zero out a block

    ssim_val = calculate_ssim(img, ref, mask)
    psnr_val = calculate_psnr(img, ref, mask)

    print("SSIM:", ssim_val.squeeze(-1).squeeze(-1))  # (B,1)
    print("PSNR:", psnr_val.squeeze(-1).squeeze(-1))  # (B,1)