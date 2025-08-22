# import necessary libraries
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from PIL import Image
import io

import torch

from model.longitudinal_recon import LongRecon
from params import ModelConfig


# load checkpoint
checkpoint_path = Path("C:/Users/user/Downloads/recon_3d_inftest/log/log_recon_250812/")
checkpoint_list = sorted(checkpoint_path.glob("*.ckpt"))
print(f"Found {len(list(checkpoint_list))} checkpoints in {checkpoint_path}")


# load model
idx = 0
checkpoint = torch.load((checkpoint_list[idx]), map_location=torch.device("cpu"))
config = ModelConfig(**checkpoint["model_config"])
model = LongRecon(device=torch.device("cpu"), modelconfig=config)
sd = checkpoint["model_state_dict"]
sd = {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}
model.load_state_dict(sd)
print(f"Loaded {idx} checkpoint, model ready.")

# load data
data_path = Path("D:/data/slabs")
data_list = sorted(data_path.glob("*.mat"))
print(f"Found {len(list(data_list))} data files in {data_path}")
data = loadmat(random.choice(data_list))
# data = loadmat(data_list[10])  # for testing, use the first file


# prepare input data
prior = torch.Tensor(data["img1_reg"]).unsqueeze(0)
prior_rot = torch.Tensor(data["img1"]).unsqueeze(0)  # can change it to data['img1'] if needed
target = torch.Tensor(data["img2"])
target = torch.stack((target, torch.zeros_like(target)), dim=-1).unsqueeze(0)
meta = torch.zeros(1, 5)

# print shapes
print(f"prior shape: {prior.shape}")
print(f"prior_rot shape: {prior_rot.shape}")
print(f"target shape: {target.shape}")
print(f"meta shape: {meta.shape}")


# inference
result, label_undersampled, mask, _, sudo_recon, prior_reg = model.long_recon(
    prior=prior,
    prior_rot=prior_rot,
    target=target,
    meta=meta,
)
result2, label_undersampled2, mask2, _, sudo_recon2, prior_reg2 = model.long_recon(
    prior=prior,
    prior_rot=prior_rot,
    target=target,
    meta=meta,
    debug=True,  # for debugging purposes
)

print(f"Result shape: {result.shape}")  # [1, 2, 16, 256, 256]
print(f"Label undersampled shape: {label_undersampled.shape}")  # [1, 16, 256, 256, 2]
print(f"Mask shape: {mask.shape}")  # [1, 1, 256, 256]
print(f"Sudo recon shape: {sudo_recon.shape}")  # [1, 1, 16, 256, 256]

result = torch.abs((result[:, 0, :, :, :] + 1j * result[:, 1, :, :, :]))
print(f"Result absolute shape: {result.shape}")  # [1, 16, 256, 256]
result2 = torch.abs((result2[:, 0, :, :, :] + 1j * result2[:, 1, :, :, :]))

z_index = 0

z_index += 1  # slice index to visualize

prior_img = prior.squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]
prior_rot_img = prior_rot.squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]
prior_reg_img = prior_reg.squeeze()[z_index].detach().cpu().numpy()  # [Z,H,W]
recon_img = result.squeeze(0)[z_index].detach().cpu().numpy()  # [2,Z,H,W] -> real
label_img = target.squeeze(0)[z_index, :, :, 0].detach().cpu().numpy()  # [Z,H,W,2] -> real
label_us_img = label_undersampled.squeeze(0)[z_index, :, :, 0].detach().cpu().numpy()  # [Z,H,W,2] -> real
mask_img = mask.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H,W]
sudo_recon_img = sudo_recon.squeeze(0).squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]


prior_img, prior_rot_img, prior_reg_img, recon_img, label_img, label_us_img, mask_img, sudo_recon_img

fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
fig.patch.set_facecolor("black")

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(prior_reg_img), cmap="gray")
plt.axis("off")
plt.title("Prior", color="white")

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(prior_img), cmap="gray")
plt.axis("off")
plt.title("Prior Registrated", color="white")

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(label_img), cmap="gray")
plt.axis("off")
plt.title("Label", color="white")

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(recon_img), cmap="gray")
plt.axis("off")
plt.title("Output", color="white")

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(abs(label_img - recon_img)), cmap="gray")
plt.axis("off")
plt.title("Error x5", color="white")

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(label_us_img), cmap="gray")
plt.axis("off")
plt.title("Undersampled Label", color="white")

from common.metric import calculate_psnr, calculate_ssim


def create_gif_from_z_slices(data_dict, output_path, title_suffix="", fps=2):
    """
    Create GIF from z-slices of 3D data

    Args:
        data_dict: Dictionary with keys as titles and values as 3D numpy arrays [Z, H, W]
        output_path: Path to save the GIF
        title_suffix: Additional suffix for the title
        fps: Frames per second for the GIF
    """
    # Get the number of z-slices (assuming all data have the same z dimension)
    z_slices = list(data_dict.values())[0].shape[0]

    frames = []

    for z_idx in range(z_slices):
        fig = plt.figure(figsize=(16, 8))
        fig.patch.set_facecolor("black")

        # Create subplots for each data type
        subplot_idx = 1
        n_cols = len(data_dict)

        for title, data in data_dict.items():
            plt.subplot(2, n_cols, subplot_idx)
            plt.imshow(np.rot90(data[z_idx]), cmap="gray")
            plt.title(f"{title} (z={z_idx})", color="white")
            plt.axis("off")
            subplot_idx += 1

        # Add difference plots in the second row if we have recon and label
        if "Reconstructed" in data_dict and "Label" in data_dict:
            subplot_idx = n_cols + 1
            for title, data in data_dict.items():
                plt.subplot(2, n_cols, subplot_idx)
                if title == "Reconstructed":
                    # Show error map
                    error = np.abs(data[z_idx] - data_dict["Label"][z_idx])
                    plt.imshow(np.rot90(error), cmap="hot")
                    plt.title(f"Error (z={z_idx})", color="white")
                elif title == "Label":
                    # Show original label again
                    plt.imshow(np.rot90(data[z_idx]), cmap="gray")
                    plt.title(f"{title} (z={z_idx})", color="white")
                else:
                    # Show other data
                    plt.imshow(np.rot90(data[z_idx]), cmap="gray")
                    plt.title(f"{title} (z={z_idx})", color="white")
                plt.axis("off")
                subplot_idx += 1

        plt.suptitle(f"Z-slice {z_idx}{title_suffix}", color="white", fontsize=16)
        plt.tight_layout()

        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="black", bbox_inches="tight", dpi=100)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        frames.append(frame)
        plt.close()

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),  # duration in milliseconds
        loop=0,
    )
    print(f"GIF saved to: {output_path}")


def create_comparison_gif(result1, result2, label, output_path, fps=2):
    """
    Create comparison GIF showing two reconstructions and ground truth
    """
    z_slices = result1.shape[0]
    frames = []

    for z_idx in range(z_slices):
        fig = plt.figure(figsize=(20, 6))
        fig.patch.set_facecolor("black")

        plt.subplot(1, 5, 1)
        plt.imshow(np.rot90(result1[z_idx]), cmap="gray")
        plt.title(f"Recon 1 (z={z_idx})", color="white")
        plt.axis("off")

        plt.subplot(1, 5, 2)
        plt.imshow(np.rot90(result2[z_idx]), cmap="gray")
        plt.title(f"Recon 2 (z={z_idx})", color="white")
        plt.axis("off")

        plt.subplot(1, 5, 3)
        plt.imshow(np.rot90(label[z_idx]), cmap="gray")
        plt.title(f"Ground Truth (z={z_idx})", color="white")
        plt.axis("off")

        plt.subplot(1, 5, 4)
        error1 = np.abs(result1[z_idx] - label[z_idx])
        plt.imshow(np.rot90(error1), cmap="hot")
        plt.title(f"Error 1 (z={z_idx})", color="white")
        plt.axis("off")

        plt.subplot(1, 5, 5)
        error2 = np.abs(result2[z_idx] - label[z_idx])
        plt.imshow(np.rot90(error2), cmap="hot")
        plt.title(f"Error 2 (z={z_idx})", color="white")
        plt.axis("off")

        plt.suptitle(f"Reconstruction Comparison - Z-slice {z_idx}", color="white", fontsize=16)
        plt.tight_layout()

        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="black", bbox_inches="tight", dpi=100)
        buf.seek(0)
        frame = Image.open(buf).convert("RGB")
        frames.append(frame)
        plt.close()

    # Save as GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"Comparison GIF saved to: {output_path}")


metric_recon_img = result
metric_recon_img2 = result2
metric_label_img = target[:, :, :, :, 0]

psnr = calculate_psnr(metric_recon_img, metric_label_img)
ssim = calculate_ssim(metric_recon_img, metric_label_img)

psnr2 = calculate_psnr(metric_recon_img2, metric_label_img)
ssim2 = calculate_ssim(metric_recon_img2, metric_label_img)

print(f"PSNR: {float(psnr):.2f}, SSIM: {float(ssim):.4f}")
print(f"PSNR2: {float(psnr2):.2f}, SSIM2: {float(ssim2):.4f}")

# Create output directory for GIFs
output_dir = Path("output_gifs")
output_dir.mkdir(exist_ok=True)

# Prepare data for GIF creation
# Convert tensors to numpy arrays for all z-slices
result_np = result.squeeze(0).detach().cpu().numpy()  # [Z, H, W]
result2_np = result2.squeeze(0).detach().cpu().numpy()  # [Z, H, W]
label_np = target.squeeze(0)[:, :, :, 0].detach().cpu().numpy()  # [Z, H, W]
prior_np = prior.squeeze(0).detach().cpu().numpy()  # [Z, H, W]
prior_rot_np = prior_rot.squeeze(0).detach().cpu().numpy()  # [Z, H, W]
prior_reg_np = prior_reg.squeeze().detach().cpu().numpy()  # [Z, H, W]
label_us_np = label_undersampled.squeeze(0)[:, :, :, 0].detach().cpu().numpy()  # [Z, H, W]
sudo_recon_np = sudo_recon.squeeze(0).squeeze(0).detach().cpu().numpy()  # [Z, H, W]

# Create comprehensive reconstruction GIF
recon_data = {"Prior": prior_np, "Prior Registered": prior_reg_np, "Reconstructed": result_np, "Label": label_np}
create_gif_from_z_slices(
    recon_data, output_dir / "reconstruction_overview.gif", title_suffix=" - Reconstruction Overview", fps=2
)

# Create detailed pipeline GIF
pipeline_data = {
    "Prior Rotated": prior_rot_np,
    "Prior Registered": prior_reg_np,
    "Sudo Recon": sudo_recon_np,
    "Final Recon": result_np,
    "Ground Truth": label_np,
    "Undersampled": label_us_np,
}
create_gif_from_z_slices(pipeline_data, output_dir / "full_pipeline.gif", title_suffix=" - Full Pipeline", fps=1.5)

# Create comparison GIF between two reconstructions
create_comparison_gif(result_np, result2_np, label_np, output_dir / "reconstruction_comparison.gif", fps=2)

# Create error analysis GIF
error1 = np.abs(result_np - label_np)
error2 = np.abs(result2_np - label_np)
error_data = {"Reconstruction 1": result_np, "Reconstruction 2": result2_np, "Error 1": error1, "Error 2": error2}
create_gif_from_z_slices(error_data, output_dir / "error_analysis.gif", title_suffix=" - Error Analysis", fps=2)

print(f"\nAll GIFs saved to: {output_dir.absolute()}")

# Display static plots for current z_index (keeping original functionality)
z_index = 1  # slice index to visualize

prior_img = prior.squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]
prior_rot_img = prior_rot.squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]
prior_reg_img = prior_reg.squeeze()[z_index].detach().cpu().numpy()  # [Z,H,W]
recon_img = result.squeeze(0)[z_index].detach().cpu().numpy()  # [2,Z,H,W] -> real
label_img = target.squeeze(0)[z_index, :, :, 0].detach().cpu().numpy()  # [Z,H,W,2] -> real
label_us_img = label_undersampled.squeeze(0)[z_index, :, :, 0].detach().cpu().numpy()  # [Z,H,W,2] -> real
mask_img = mask.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H,W]
sudo_recon_img = sudo_recon.squeeze(0).squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]

prior_img, prior_rot_img, prior_reg_img, recon_img, label_img, label_us_img, mask_img, sudo_recon_img

fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
fig.patch.set_facecolor("black")

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(prior_reg_img), cmap="gray")
plt.axis("off")
plt.title("Prior", color="white")

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(prior_img), cmap="gray")
plt.axis("off")
plt.title("Prior Registrated", color="white")

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(label_img), cmap="gray")
plt.axis("off")
plt.title("Label", color="white")

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(recon_img), cmap="gray")
plt.axis("off")
plt.title("Output", color="white")

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(abs(label_img - recon_img)), cmap="gray")
plt.axis("off")
plt.title("Error x5", color="white")

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(label_us_img), cmap="gray")
plt.axis("off")
plt.title("Undersampled Label", color="white")


def continuous_brain_mask(img, thr=0.1):
    mask = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        row = img[i]
        idx = np.where(row > thr * img.max())[0]
        if idx.size > 0:
            mask[i, idx.min() : idx.max() + 1] = 1
    return mask


masks = []
for i in range(metric_label_img.shape[1]):
    masks.append(continuous_brain_mask(metric_label_img[0, i, :, :], thr=0.1))

# plot each masks
plt.figure(figsize=(3 * len(masks), 9))
for i, mask in enumerate(masks):
    plt.subplot(3, len(masks), i + 1)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
for i in range(len(masks)):
    plt.subplot(3, len(masks), len(masks) + i + 1)
    plt.imshow(metric_recon_img[0, i, :, :], cmap="gray")
    plt.axis("off")
for i in range(len(masks)):
    plt.subplot(3, len(masks), 2 * len(masks) + i + 1)
    plt.imshow(metric_label_img[0, i, :, :], cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()

mask = torch.Tensor(masks).unsqueeze(0)

psnr = calculate_psnr(metric_recon_img, metric_label_img, mask)
ssim = calculate_ssim(metric_recon_img, metric_label_img, mask)
f = ratio = (mask == 1).float().mean().item()
ssim = (ssim - (1 - f)) / f

print(f"PSNR: {float(psnr):.2f}, SSIM: {float(ssim):.4f}")


z_index += 0  # slice index to visualize

prior_img = prior.squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]
prior_rot_img = prior_rot.squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]
prior_reg_img = prior_reg.squeeze()[z_index].detach().cpu().numpy()  # [Z,H,W]
recon_img2 = result2.squeeze(0)[z_index].detach().cpu().numpy()  # [2,Z,H,W] -> real
label_img = target.squeeze(0)[z_index, :, :, 0].detach().cpu().numpy()  # [Z,H,W,2] -> real
label_us_img = label_undersampled.squeeze(0)[z_index, :, :, 0].detach().cpu().numpy()  # [Z,H,W,2] -> real
mask_img = mask.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H,W]
sudo_recon_img = sudo_recon.squeeze(0).squeeze(0)[z_index].detach().cpu().numpy()  # [Z,H,W]

# Plot 2x4 grid
plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.imshow(np.rot90(prior_img), cmap="gray")
plt.title("Prior")
plt.axis("off")
plt.subplot(2, 4, 2)
plt.imshow(np.rot90(prior_rot_img), cmap="gray")
plt.title("Prior Rotated")
plt.axis("off")
plt.subplot(2, 4, 3)
plt.imshow(np.rot90(prior_reg_img), cmap="gray")
plt.title("Prior Registered")
plt.axis("off")
plt.subplot(2, 4, 4)
plt.imshow(np.rot90(recon_img2), cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(np.rot90(label_img), cmap="gray")
plt.title("Label (Target)")
plt.axis("off")
plt.subplot(2, 4, 6)
plt.imshow(np.rot90(label_us_img), cmap="gray")
plt.title("Label Undersampled")
plt.axis("off")
plt.subplot(2, 4, 7)
plt.imshow(mask_img, cmap="gray")
plt.title("Mask")
plt.axis("off")
plt.subplot(2, 4, 8)
plt.imshow(np.rot90(sudo_recon_img), cmap="gray")
plt.title("Sudo Recon")
plt.axis("off")

plt.tight_layout()
plt.show()


loss = torch.nn.functional.mse_loss(torch.Tensor(recon_img), torch.Tensor(label_img))
print(f"Loss: {loss.item()}")  # Print the loss value

loss = torch.nn.functional.mse_loss(torch.Tensor(sudo_recon_img), torch.Tensor(label_img))
print(f"Sudo Recon Loss: {loss.item()}")  # Print the sudo recon loss value

loss2 = torch.nn.functional.mse_loss(torch.Tensor(recon_img2), torch.Tensor(label_img))
print(f"Loss2: {loss2.item()}")  # Print the loss value for the second reconstruction
