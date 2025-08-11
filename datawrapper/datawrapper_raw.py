import glob
from dataclasses import dataclass

import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


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
    Target = 1
    Mask = 2
    Time = 3
    Meta = 4
    Name = 5

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
                self.file_list = self.file_list

        # Set keys for data
        self.prior_key = loader_cfg.prior_key
        self.target_key = loader_cfg.target_key
        self.target_mask_key = loader_cfg.target_mask_key

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

        target_raw_img = torch.from_numpy(file_mat[self.target_key + "_raw_img"]).type(torch.float32)
        target_raw_sen = torch.from_numpy(file_mat[self.target_key + "_raw_sen"]).type(torch.float32)
        target_raw_img_complex = target_raw_img[:, :, 0, :, :] + 1j * target_raw_img[:, :, 1, :, :]
        target_raw_sen_complex = target_raw_sen[:, :, 0, :, :] + 1j * target_raw_sen[:, :, 1, :, :]
        target = target_raw_img_complex * torch.conj(target_raw_sen_complex)
        target = target.sum(dim=1, keepdim=False)

        mask = torch.from_numpy(file_mat[self.target_mask_key]).type(torch.float)
        mask = mask[mask.shape[0] // 2 : mask.shape[0] // 2 + 1, :, :]

        if self.prior_key in file_mat:
            prior = torch.from_numpy(file_mat[self.prior_key]).type(torch.float)
        else:
            prior = torch.zeros_like(target, dtype=torch.float)

        target = torch.stack([target.real, target.imag], dim=-1).type(torch.float32)

        meta = self._get_meta(file_mat=file_mat)
        time = torch.randint(low=0, high=self.num_timesteps, size=[1])

        _name = self.file_list[idx].split("/")[-1]

        return (
            prior,
            target,
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
) -> tuple[
    DataLoader,
    DataWrapper,
    int,
]:
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
        pin_memory=False,
        persistent_workers=True,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
        len(dataset),
    )
