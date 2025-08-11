import random

import torch

from common.logger import logger

FIELD_MAX = 2.0
NOISE_MAX = 0.02


class PhaseAugment:
    def __init__(
        self,
    ) -> None:
        logger.info(f"PhaseAugment initialized with FIELD_MAX={FIELD_MAX}, NOISE_MAX={NOISE_MAX}")

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        bias = random.uniform(0, torch.pi * 2)

        H, W = target.shape[-2:]
        x = torch.arange(W, dtype=torch.float32) / (W - 1) * torch.pi * 2
        y = torch.arange(H, dtype=torch.float32) / (H - 1) * torch.pi * 2
        x, y = torch.meshgrid(x, y, indexing="ij")
        x = x * random.uniform(0.0, 1.0)
        y = y * random.uniform(0.0, 1.0)
        field = (x + y) / (x + y).max() * 2 * torch.pi
        field = field * random.uniform(0.0, FIELD_MAX)

        noise = torch.randn_like(x) * random.uniform(0, NOISE_MAX) * torch.pi

        phase = bias + field + noise
        phase = torch.exp(1j * phase)

        if target.dim() == 3:
            phase = phase.unsqueeze(0)
        elif target.dim() != 4:
            phase = phase.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 5:
            phase = phase.squeeze(0)
        else:
            raise ValueError(f"Invalid target shape: {target.shape}")

        target = target * phase
        return target
