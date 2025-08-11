import torch
from model.backbone.unet import Unet
from torch import nn


class SudoreconNet(nn.Module):
    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        in_chans: int = 1,
        out_chans: int = 1,
    ):
        super().__init__()
        self.net = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=pools,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.net(
            x=x,
        )
        return output
