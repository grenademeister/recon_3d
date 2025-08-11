import numpy as np
import torch
from scipy.ndimage import map_coordinates


def rotate_and_translate_with_scipy(
    img: torch.Tensor,
    angle: float,
    tx: float,
    ty: float,
) -> torch.Tensor:
    if img.dim() != 3 and img.dim() != 4:
        raise NotImplementedError("img shape has to be 3D or 4D")

    img = img.cpu()

    angle = angle * 10 * torch.pi / 180
    tx = tx * 20
    ty = ty * 20

    is_4d = img.ndim == 4

    if is_4d:
        batch, depth, height, width = img.shape
    else:
        depth, height, width = img.shape

    center_x = width / 2
    center_y = height / 2

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    theta = np.array(
        [
            [
                cos_theta,
                -sin_theta,
                -center_x * cos_theta + center_y * sin_theta + center_x + tx,
            ],
            [
                sin_theta,
                cos_theta,
                -center_x * sin_theta - center_y * cos_theta + center_y + ty,
            ],
        ]
    )

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coords = np.stack([x, y, np.ones_like(x)], axis=0)
    transformed_coords = np.tensordot(theta, coords.reshape(3, -1), axes=1).reshape(2, height, width)

    if is_4d:
        output = np.zeros_like(img, dtype=np.float32)
        for b in range(batch):
            for z in range(depth):
                output[b, z, ...] = map_coordinates(img[b, z, ...], transformed_coords, order=5, mode="nearest")
    else:
        output = np.zeros_like(img, dtype=np.float32)
        for z in range(depth):
            output[z, ...] = map_coordinates(img[z, ...], transformed_coords, order=5, mode="nearest")

    return torch.from_numpy(output)


def rotate_and_translate_3d_with_scipy(
    img: torch.Tensor,
    angle_x: float,
    angle_y: float,
    angle_z: float,
    tx: float,
    ty: float,
    tz: float,
) -> torch.Tensor:
    """3D rotate (about x,y,z) and translate (x,y,z) a 3D volume using scipy interpolation.

    Args:
        img: (D,H,W) or (B,D,H,W) tensor.
        angle_x, angle_y, angle_z: rotation parameters (scaled like the 2D version: *10 deg).
        tx, ty, tz: translation parameters (scaled like 2D version: *20 voxels) along x (W), y (H), z (D).

    Returns:
        Transformed tensor with same shape as input.
    """
    if img.dim() not in (3, 4):
        raise NotImplementedError("img shape has to be 3D or 4D: (D,H,W) or (B,D,H,W)")

    img = img.cpu()

    # Scale to degrees then radians (matching 2D helper style) & translations
    ax = float(angle_x) * 10 * torch.pi / 180
    ay = float(angle_y) * 10 * torch.pi / 180
    az = float(angle_z) * 10 * torch.pi / 180
    tx = float(tx) * 20
    ty = float(ty) * 20
    tz = float(tz) * 20

    is_4d = img.ndim == 4
    if is_4d:
        batch, depth, height, width = img.shape
    else:
        depth, height, width = img.shape

    # Use (N-1)/2 so that the central voxel maps onto itself (avoids half-voxel shift)
    cz = (depth - 1) / 2.0
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    # Rotation matrices (right-handed). Combined R = Rz * Ry * Rx
    sx, cx_ = np.sin(ax), np.cos(ax)
    sy, cy_ = np.sin(ay), np.cos(ay)
    sz, cz_ = np.sin(az), np.cos(az)

    Rx = np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]], dtype=np.float64)
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]], dtype=np.float64)
    Rz = np.array([[cz_, -sz, 0], [sz, cz_, 0], [0, 0, 1]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    R_inv = R.T  # rotation inverse (orthonormal)

    t = np.array([tz, ty, tx], dtype=np.float64)  # order matches (z,y,x)
    center = np.array([cz, cy, cx], dtype=np.float64)[:, None]

    # Create output grid coordinates (p_out)
    z, y, x = np.meshgrid(
        np.arange(depth, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    grid = np.stack([z, y, x], axis=0).reshape(3, -1)  # (3, N)

    # Inverse mapping: p_in = R_inv @ (p_out - center - t) + center
    p_in = R_inv @ (grid - center) - R_inv @ t[:, None] + center
    coords = p_in.reshape(3, depth, height, width)

    # Sample
    if is_4d:
        out = np.zeros_like(img, dtype=np.float32)
        for b in range(batch):
            out[b] = map_coordinates(img[b].numpy(), coords, order=5, mode="nearest")
    else:
        out = map_coordinates(
            img.numpy(),
            coords,
            order=5,
            mode="nearest",
        ).astype(np.float32)

    return torch.from_numpy(out)

if __name__ == "__main__":
    # Example usage
    from scipy.io import loadmat
    data = loadmat("14409682_0.mat")["target"]
    img = torch.Tensor(data)  # Example 3D image
    img = img.unsqueeze(0)  # Add batch dimension if needed
    img[0, 126:130, : , :] = 1  # Set a slice to 1 for visibility
    img[0, :, 126:130, :] = 1  # Set another slice to 1 for visibility
    img[0, :, :, 126:130] = 1  # Set another slice to 1 for visibility

    rot_x, rot_y, rot_z = 0.2, 0.1, -0.1  # Example rotation angles
    tx, ty, tz = 0.5, 0.2, -0.3  # Example translations
    # Rotate and translate the image
    rotated_img = rotate_and_translate_3d_with_scipy(img, rot_x, rot_y, rot_z, tx, ty, tz)
    print(rotated_img.shape)  # Should match input shape
    rotated_back = rotate_and_translate_3d_with_scipy(rotated_img, -rot_x, -rot_y, -rot_z, -tx, -ty, -tz)

    data1 = img.numpy()[0, :, :, :]  # Convert to numpy for visualization
    data2 = rotated_img.numpy()[0, :, :, :]  # For demonstration, using the same data
    data3 = rotated_back.numpy()[0, :, :, :]  # Should match original data
        
    # Shared contrast bounds using percentiles across volumes per slice
    def get_clip_bounds(slice_idx):
        slices = [
            data1[:, :, slice_idx],
            data2[:, :, slice_idx],
            data3[:, :, slice_idx],
        ]
        all_vals = np.concatenate([s.flatten() for s in slices])
        p1, p99 = np.percentile(all_vals, [1, 99])
        return p1, p99

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # Visualization with percentile clipping
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    initial_slice = 0
    vmin, vmax = get_clip_bounds(initial_slice)
    im1 = ax1.imshow(data1[:, :, initial_slice], cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("Prior")
    im2 = ax2.imshow(data2[:, :, initial_slice], cmap="gray", vmin=vmin, vmax=vmax)
    ax2.set_title("Prior Reg")
    im3 = ax3.imshow(data3[:, :, initial_slice], cmap="gray", vmin=vmin, vmax=vmax)
    ax3.set_title("Target")

    fig.suptitle(f"Slice {initial_slice}/{data1.shape[2]-1}")

    # Slider over common valid range
    max_slices = min(data1.shape[2], data2.shape[2], data3.shape[2]) - 1
    ax_slider = plt.axes([0.2, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, "Slice", 0, max_slices, valinit=0, valfmt="%d")


    def update(val):
        slice_idx = int(slider.val)
        vmin, vmax = get_clip_bounds(slice_idx)
        im1.set_data(data1[:, :, slice_idx])
        im2.set_data(data2[:, :, slice_idx])
        im3.set_data(data3[:, :, slice_idx])
        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)
        im3.set_clim(vmin, vmax)
        fig.suptitle(f"Slice {slice_idx}/{data1.shape[2]-1}")
        fig.canvas.draw_idle()


    slider.on_changed(update)
    plt.show()

        