import torch
from model.backbone.unet import Unet
from torch import nn


def _ifft2c(img_k: torch.Tensor) -> torch.Tensor:
    img = torch.fft.ifftn(torch.fft.ifftshift(img_k, dim=(-2, -1)), dim=(-2, -1))
    return img


def _fft2c(img: torch.Tensor) -> torch.Tensor:
    img_k = torch.fft.fftshift(torch.fft.fftn(img, dim=(-2, -1)), dim=(-2, -1))
    return img_k


class VarNetBlock(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        model_output = self.model(x)

        consistency_term = _ifft2c(_fft2c(x0 - model_output) * mask) * self.dc_weight
        output = model_output + consistency_term
        return torch.abs(output)


class VarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        in_chans: int = 1,
        out_chans: int = 1,
    ):
        super().__init__()
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    Unet(
                        in_chans=in_chans,
                        out_chans=out_chans,
                        chans=chans,
                        num_pool_layers=pools,
                    )
                )
                for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        output = x

        for cascade in self.cascades:
            output = cascade(output, x0, mask)

        return output
