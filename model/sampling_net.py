import math

import torch
from torch import Tensor, nn
from torch.nn import functional


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.t: nn.Parameter = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def forward(self) -> Tensor:
        t = self.t
        half_dim = self.dim
        emb = math.log(100) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.sin(torch.flip(emb, dims=[1]))
        return emb


class SamplingNet(nn.Module):
    def __init__(
        self,
        chans: int = 32,
        conv1d_kernel: int = 3,
        masknet_num_first_layers: int = 16,
    ) -> None:
        super().__init__()
        conv1d_pad = (conv1d_kernel - 1) // 2
        init_vector_dim: int = 32
        linear_dim: int = 64

        self.embedding = TimeEmbedding(init_vector_dim)

        self.linear = nn.Sequential(
            nn.Linear(init_vector_dim, linear_dim),
            nn.SiLU(inplace=True),
            nn.Linear(linear_dim, linear_dim),
        )

        layers = []
        layers.append(nn.Conv1d(1, chans, kernel_size=conv1d_kernel, padding=conv1d_pad))
        layers.append(nn.SiLU(inplace=True))
        layers.append(nn.GroupNorm(min(4, chans), chans))

        for _ in range(masknet_num_first_layers):
            layers.append(nn.Conv1d(chans, chans, kernel_size=conv1d_kernel, padding=conv1d_pad))
            layers.append(nn.SiLU(inplace=True))
            layers.append(nn.GroupNorm(4, chans))

        layers.append(nn.Conv1d(chans, 1, kernel_size=conv1d_kernel, padding=conv1d_pad))

        self.last_layer = nn.Sequential(*layers)

        self.eps = nn.Parameter(torch.tensor(1e-6), requires_grad=False)
        self.clip_val = nn.Parameter(torch.tensor(3.5), requires_grad=False)

    def forward(
        self,
        mask_len: torch.Tensor,
    ) -> torch.Tensor:
        output = self.embedding()
        output = output.unsqueeze(dim=1)  # [B x 1 x D]
        output = self.linear(output)

        output = self.last_layer(output)

        half_len = (mask_len + 1) // 2
        output = functional.interpolate(
            output,
            size=half_len,
            mode="linear",
            align_corners=False,
        )

        output = output.squeeze(dim=1)
        output = torch.cat((output, torch.flip(output, dims=[1])), dim=1)

        output = torch.tanh(output / self.clip_val) * self.clip_val

        std = torch.std(output, dim=1, keepdim=True)
        if std >= 1:
            output = output / (std + self.eps)

        return torch.sigmoid(output)
