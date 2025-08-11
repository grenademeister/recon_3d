import glob
import random
from dataclasses import dataclass

import numpy as np
import torch
from datawrapper.phase_augment import PhaseAugment
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from components.registration import rotate_and_translate_with_scipy, rotate_and_translate_3d_with_scipy

prob_flip: float = 0.5
scale_fac: float = 0.05


@dataclass
class LoaderConfig:
    data_type: str
    batch: int
    num_workers: int
    shuffle: bool
    debug_mode: bool
    rotation_conf: str
    prior_key: str
    target_key: str
    target_mask_key: str
    use_meta: bool


class DataWrapper(Dataset):
    Prior = 0
    PriorRot = 1
    RegLabel = 2
    Target = 3
    Mask = 4
    Time = 5
    Meta = 6
    Name = 7

    def __init__(
        self,
        file_path: list[str],
        training_mode: bool,
        loader_cfg: LoaderConfig,
    ) -> None:
        self.num_timesteps: int = 1000

        # Initialize the dataset
        super().__init__()
        total_list: list[str] = []
        for _file_path in file_path:
            total_list += glob.glob(f"{_file_path}/{loader_cfg.data_type}")
        self.file_list = total_list

        if len(self.file_list) == 0:
            raise ValueError(f"No files found in the specified path: {file_path}. Please check the path and data type.")

        # Set training mode and debug mode
        self.training_mode = training_mode
        self.use_meta = loader_cfg.use_meta

        if loader_cfg.debug_mode:
            if training_mode:
                self.file_list = self.file_list[:500]
            else:
                self.file_list = self.file_list[:100]

        else:
            if training_mode:
                self.file_list = self.file_list
            else:
                # self.file_list = self.file_list
                self.file_list = self.file_list[::10]

        # Set rotation configuration
        rot_conf = loader_cfg.rotation_conf.split("_")
        if len(rot_conf) == 2:
            self.rot_type = rot_conf[0]
            self.rot_angle = int(rot_conf[1])
        else:
            self.rot_type = "none"
            self.rot_angle = 0
        if self.rot_type not in ["none", "rand", "fixed"]:
            raise ValueError(f"Invalid rotation type: {self.rot_type}. " "Choose from 'none', 'random', or 'fixed'.")

        # Set keys for data
        self.prior_key = loader_cfg.prior_key
        self.target_key = loader_cfg.target_key
        self.target_mask_key = loader_cfg.target_mask_key

        self.phase_augment = PhaseAugment()

    def _gen_rot_mat_dep(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a random rotation matrix for 3D images.
        This is a deprecated version of rotation matrix generation
        only capable of 2D rotation
        """
        reg_ran = torch.rand(3)
        reg_scale = 10
        if self.rot_type == "none":
            reg_mat = torch.tensor([0.0, 0.0, 0.0])
            img1_rot = img
        elif self.rot_type == "rand":
            reg_mat = (self.rot_angle * reg_ran - self.rot_angle * (1 - reg_ran)) / reg_scale
            img1_rot = rotate_and_translate_with_scipy(img=img, angle=reg_mat[0], tx=reg_mat[1], ty=reg_mat[2])
        elif self.rot_type == "fixed":
            reg_mat = (self.rot_angle * reg_ran + self.rot_angle * (1 - reg_ran)) / reg_scale
            img1_rot = rotate_and_translate_with_scipy(img=img, angle=reg_mat[0], tx=reg_mat[1], ty=reg_mat[2])
        else:
            raise ValueError(f"Invalid rotation type: {self.rot_type}. " "Choose from 'none', 'random', or 'fixed'.")
        return img1_rot, reg_mat
    
    def _gen_rot_mat(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a random rotation matrix for 3D images.
        This version supports 3D rotation and translation.
        """
        reg_ran = torch.rand(6)
        reg_scale = 10
        if self.rot_type == "none":
            reg_mat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            img1_rot = img
        elif self.rot_type == "rand":
            reg_mat = (self.rot_angle * reg_ran - self.rot_angle * (1 - reg_ran)) / reg_scale
            img1_rot = rotate_and_translate_3d_with_scipy(img=img, angle_x=reg_mat[0], angle_y=reg_mat[1], angle_z=reg_mat[2], tx=reg_mat[3], ty=reg_mat[4], tz=reg_mat[5])
        elif self.rot_type == "fixed":
            reg_mat = (self.rot_angle * reg_ran + self.rot_angle * (1 - reg_ran)) / reg_scale
            img1_rot = rotate_and_translate_3d_with_scipy(img=img, angle_x=reg_mat[0], angle_y=reg_mat[1], angle_z=reg_mat[2], tx=reg_mat[3], ty=reg_mat[4], tz=reg_mat[5])
        else:
            raise ValueError(f"Invalid rotation type: {self.rot_type}. " "Choose from 'none', 'random', or 'fixed'.")
        return img1_rot, reg_mat

    def _get_meta(
        self,
        file_mat: dict,
    ) -> torch.Tensor:
        """
        meta_keys = [
            "ScanOptions",
            "FlipAngle",
            "EchoTime",
            "RepetitionTime",
            "InversionTime",
            # "EchoTrainLength",
            # "WaterFatShift",
        ]
        """
        if "meta" in file_mat:
            return torch.tensor(file_mat["meta"], dtype=torch.float32).squeeze()

        meta: list[float] = [
            file_mat.get("ScanOptions", 0),
            file_mat.get("FlipAngle", 0),
            file_mat.get("EchoTime", 0),
            file_mat.get("RepetitionTime", 0),
            file_mat.get("InversionTime", 0),
            # file_mat.get("EchoTrainLength", 0),
            # file_mat.get("WaterFatShift", 0),
        ]
        meta_tensor = torch.zeros(len(meta), dtype=torch.float32) if not self.use_meta else torch.tensor(meta, dtype=torch.float32).squeeze()
        return meta_tensor

    def __getitem__(
        self,
        idx: int,
    ):
        file_mat = loadmat(self.file_list[idx])

        target = torch.from_numpy(file_mat[self.target_key]).type(torch.complex64)
        mask = torch.ones_like(target, dtype=torch.float32)
        mask = mask[mask.shape[0] // 2 : mask.shape[0] // 2 + 1, :, :]

        if self.prior_key in file_mat:
            prior = torch.from_numpy(file_mat[self.prior_key]).type(torch.float)
        else:
            prior = torch.zeros_like(target, dtype=torch.float)

        if self.training_mode:
            if random.random() > prob_flip:
                prior = torch.flip(prior, dims=[1])
                target = torch.flip(target, dims=[1])
            if random.random() > prob_flip:
                prior = torch.flip(prior, dims=[2])
                target = torch.flip(target, dims=[2])

            prior = (1 + np.random.normal() * scale_fac) * prior
            target = (1 + np.random.normal() * scale_fac) * target

            target = self.phase_augment(target)

        target = torch.stack([target.real, target.imag], dim=-1).type(torch.float32)
        prior_rot, reg_mat = self._gen_rot_mat(img=prior)

        meta = self._get_meta(file_mat=file_mat)
        time = torch.randint(low=0, high=self.num_timesteps, size=[1])

        _name = self.file_list[idx].split("/")[-1]

        return (
            prior, # shape: [Z, H, W]
            prior_rot,
            reg_mat,
            target, # shape: [Z, H, W, 2]
            mask,
            time,
            meta,
            _name,
        )

    def __len__(self) -> int:
        return len(self.file_list)


def get_data_wrapper_loader(
    file_path: list[str],
    training_mode: bool,
    loader_cfg: LoaderConfig,
) -> tuple[DataLoader, DataWrapper, int]:
    dataset = DataWrapper(
        file_path=file_path,
        training_mode=training_mode,
        loader_cfg=loader_cfg,
    )

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
        len(dataset),
    )
