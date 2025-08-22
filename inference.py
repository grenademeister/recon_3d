import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import torch

from model.longitudinal_recon import LongRecon
from params import ModelConfig

checkpoint_path = Path("C:/Users/user/Downloads/recon_3d/log/log_recon_250812/00000_train/checkpoints")
checkpoint_list = sorted(checkpoint_path.glob("*.ckpt"))
print(f"Found {len(list(checkpoint_list))} checkpoints in {checkpoint_path}")

idx = 1
checkpoint = torch.load((checkpoint_list[idx]), map_location=torch.device('cpu'))
config = ModelConfig(**checkpoint['model_config'])
model = LongRecon(device=torch.device('cpu'), modelconfig=config)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded {idx} checkpoint, model ready.")

data_path = Path("D:/data/slabs")
data_list = sorted(data_path.glob("*.mat"))
print(f"Found {len(list(data_list))} data files in {data_path}")
data = loadmat(random.choice(data_list))

prior = torch.Tensor(data['img1_reg']).unsqueeze(0)
prior_rot = torch.Tensor(data['img1_reg']).unsqueeze(0)  # can change it to data['img1'] if needed
target = torch.Tensor(data['img2'])
target = torch.stack((target, torch.zeros_like(target)), dim=-1).unsqueeze(0)
meta = torch.zeros(1,5)

result, label_undersampled, mask, _, sudo_recon, prior_reg = model.long_recon(
    prior=prior,
    prior_rot=prior_rot,
    target=target,
    meta=meta,
)
print(f"Result shape: {result.shape}") # [1, 2, 16, 256, 256]
print(f"Label undersampled shape: {label_undersampled.shape}") # [1, 16, 256, 256, 2]
print(f"Mask shape: {mask.shape}") # [1, 1, 256, 256]
print(f"Sudo recon shape: {sudo_recon.shape}") # [1, 1, 16, 256, 256]

plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(prior.squeeze().numpy()[8,:,:], cmap='gray')
plt.title('Prior Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(result.squeeze().numpy()[0,8,:,:], cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(label_undersampled.squeeze().numpy()[8,:,:,0], cmap='gray')
plt.title('Label Undersampled')
plt.axis('off')
plt.tight_layout()
plt.show()