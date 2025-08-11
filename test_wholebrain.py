import argparse
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from core_funcs import get_loss_func, log_summary
from datawrapper.datawrapper_wholebrain import DataWrapper, LoaderConfig, get_data_wrapper_loader
from model.longitudinal_recon import LongRecon
from scipy.io import savemat
from torch import Tensor
from torch.utils.data import DataLoader

from common.logger import logger, logger_add_handler
from common.metric import calculate_psnr, calculate_ssim
from common.utils import call_next_id, separator, validate_tensor_dimensions, validate_tensors
from common.wrapper import error_wrap
from components.metriccontroller import MetricController
from components.wholebrain_recon import longitudinal_recon_wholebrain
from params import ModelConfig

warnings.filterwarnings("ignore")


default_root: str = "/home/juhyung/data/longitudinal/data/data_longitudinal_test_snuhosptial"
default_run_dir: str = "/home/juhyung/data/longitudinal/log/log_recon_wholebrain_2025_07_13"

DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)
RUN_DIR: str = os.environ.get("RUN_DIR", default_run_dir)
TEST_DATASET: list[str] = [
    DATA_ROOT,
]


@dataclass
class TestConfig:
    # Dataset
    longitudinal_checkpoints: str = "/home/juhyung/data/longitudinal/log/log_recon_2025_07_13/00001_train/checkpoints/checkpoint_20.ckpt"

    test_dataset: list[str] = field(default_factory=lambda: TEST_DATASET)
    data_type: str = "*.mat"
    debugmode: bool = False

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = Path(RUN_DIR)
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["longrecon"] = "longrecon"

    # Test params
    gpu: str = "7"
    valid_batch: int = 1
    num_workers: int = 4
    device: torch.device | None = None
    loss_model: Literal["l1", "l2"] = "l2"

    # hyper
    use_meta: bool = False
    rotation_conf: str = "none_0"
    prior_key: str = "img1_reg"
    target_key: str = "img2"
    target_mask_key: str = "mask2"


parser = argparse.ArgumentParser(description="Test Configuration")
test_dict = asdict(TestConfig())
for key, default_value in test_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )
args = parser.parse_args()

NET_LONGITUDINAL = LongRecon | torch.nn.DataParallel[LongRecon]


def test_part_longrecon(
    _data: dict[DataWrapper, Tensor | str],
    test_dir: Path,
    network_longitudinal: NET_LONGITUDINAL,
    save_val: bool,
    test_state: MetricController,
    config: TestConfig,
) -> None:
    loss_func = get_loss_func(loss_model=config.loss_model)

    prior: Tensor = _data[DataWrapper.Prior].to(config.device)
    target: Tensor = _data[DataWrapper.Target].to(config.device)
    target_mask: Tensor = _data[DataWrapper.Mask].to(config.device)
    meta: Tensor = _data[DataWrapper.Meta].to(config.device)
    name = _data[DataWrapper.Name][0]

    logger.info(f"Testing {name} started...")

    validate_tensors([prior, target, target_mask])
    validate_tensor_dimensions([prior], 4)  # [B, Z, H, W]
    validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C]

    batch_cnt = target.shape[0]
    if batch_cnt != 1:
        raise ValueError("Batch size must be 1 for whole brain reconstruction.")
    (
        output,
        mask,
        mask_prob,
        sudorecon,
        prior_reg,
    ) = longitudinal_recon_wholebrain(
        long_recon=network_longitudinal,
        prior=prior,
        target=target,
        meta=meta,
        batch_size=config.valid_batch,
    )
    validate_tensors([output])
    validate_tensor_dimensions([output], 5)  # [B, Z, H, W, C]

    output_abs = torch.abs(output[:, :, :, :, 0] + 1j * output[:, :, :, :, 1])  # [B, Z, H, W]
    target_abs = torch.abs(target[:, :, :, :, 0] + 1j * target[:, :, :, :, 1])  # [B, Z, H, W]

    loss = loss_func(output_abs, target_abs)
    loss = torch.mean(loss, dim=(1, 2, 3), keepdim=True)
    test_state.add("loss", loss)

    for slice in range(output.shape[1]):
        mask_slice = target_mask[:, slice : slice + 1, :, :]
        output_slice = output_abs[:, slice : slice + 1, :, :]
        target_slice = target_abs[:, slice : slice + 1, :, :]
        if torch.any(mask_slice):
            test_state.add("psnr", calculate_psnr(output_slice, target_slice, mask_slice))
            test_state.add("ssim", calculate_ssim(output_slice, target_slice, mask_slice))

    if not save_val:
        return

    os.makedirs(test_dir, exist_ok=True)
    save_dict = {
        "prior": prior.cpu().detach().numpy()[0, ...],
        "out": output.cpu().detach().numpy()[0, ...],
        "target": target.cpu().detach().numpy()[0, ...],
        "mask": mask.cpu().detach().numpy()[0, ...],
        "brain_mask": target_mask.cpu().detach().numpy()[0, ...],
    }
    if sudorecon is not None:
        save_dict["sudo_recon"] = sudorecon.cpu().detach().numpy()[0, ...]
    if mask_prob is not None:
        save_dict["mask_prob"] = mask_prob.cpu().detach().numpy()[0, ...]
    if prior_reg is not None:
        save_dict["prior_reg"] = prior_reg.cpu().detach().numpy()[0, ...]
    savemat(f"{test_dir}/{name}_res.mat", save_dict)


def test_part(
    valid_state: MetricController,
    valid_loader: DataLoader,
    network_longitudinal: NET_LONGITUDINAL,
    run_dir: Path,
    save_val: bool,
    config: TestConfig,
) -> float:
    if config.device is None:
        raise TypeError("device is not to be None")

    network_longitudinal.eval()

    for _data in valid_loader:
        if config.model_type in ("longrecon"):
            network_longitudinal.set_fixed_thresh()
            test_part_longrecon(
                _data=_data,
                test_dir=run_dir / "test",
                network_longitudinal=network_longitudinal,
                save_val=save_val,
                test_state=valid_state,
                config=config,
            )
        else:
            raise KeyError("model type not matched")

    log_summary(state=valid_state, log_std=True, init_time=config.init_time)

    primary_metric = valid_state.mean("loss")
    return primary_metric


class Tester:
    run_dir: Path
    network_longitudinal: NET_LONGITUDINAL
    test_loader: DataLoader
    config: TestConfig
    modelconfig: ModelConfig

    def __init__(
        self,
    ) -> None:
        self.config = TestConfig()
        for key, value in vars(args).items():
            if value is not None and hasattr(self.config, key):
                if isinstance(getattr(self.config, key), bool):
                    setattr(self.config, key, bool(value))
                else:
                    setattr(self.config, key, value)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu

        self.config.init_time = time.time()
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dir setting
        self.run_dir = self.config.run_dir / f"{call_next_id(self.config.run_dir):05d}_test"
        logger_add_handler(logger, f"{self.run_dir/'log.log'}", self.config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=True)

        # log config
        logger.info(separator())
        logger.info("Text Config")
        config_dict = asdict(self.config)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

    def __call__(
        self,
    ) -> None:
        self._set_data()
        self._set_network()
        self._test()

    @error_wrap
    def _set_data(
        self,
    ) -> None:
        logger.info(separator())
        test_loader_cfg = LoaderConfig(
            data_type=self.config.data_type,
            batch=self.config.valid_batch,
            num_workers=self.config.num_workers,
            shuffle=False,
            debug_mode=self.config.debugmode,
            rotation_conf=self.config.rotation_conf,
            prior_key=self.config.prior_key,
            target_key=self.config.target_key,
            target_mask_key=self.config.target_mask_key,
            use_meta=self.config.use_meta,
        )
        self.test_loader, _, test_len = get_data_wrapper_loader(
            file_path=self.config.test_dataset,
            training_mode=False,
            loader_cfg=test_loader_cfg,
        )
        logger.info(f"Test dataset length : {test_len}")

    @error_wrap
    def _set_network(
        self,
    ) -> None:
        longitudinal_checkpoint_data = torch.load(
            self.config.longitudinal_checkpoints,
            map_location="cpu",
            weights_only=True,
        )

        if not (("model_state_dict" in longitudinal_checkpoint_data) and ("model_config" in longitudinal_checkpoint_data)):
            logger.error("Invalid Checkpoint")
            raise KeyError("Invalid Checkpoint")

        self.modelconfig = ModelConfig(**longitudinal_checkpoint_data["model_config"])
        self.network_longitudinal = LongRecon(device=self.config.device, modelconfig=self.modelconfig)
        load_state_dict = longitudinal_checkpoint_data["model_state_dict"]

        _state_dict = {}
        for key, value in load_state_dict.items():
            new_key = key.replace("module.", "")
            _state_dict[new_key] = value

        try:
            self.network_longitudinal.load_state_dict(_state_dict, strict=True)
        except Exception as err:
            logger.warning(f"Strict load failure. Trying to load weights available: {err}")
            self.network_longitudinal.load_state_dict(_state_dict, strict=False)

        logger.info(separator())
        logger.info("Model Config")
        config_dict = asdict(self.modelconfig)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

        self.network_longitudinal = self.network_longitudinal.to(self.config.device)

    @error_wrap
    def _test(self) -> None:
        test_state = MetricController()
        test_state.reset()
        logger.info(separator())
        logger.info("Test")
        with torch.no_grad():
            test_part(
                valid_state=test_state,
                valid_loader=self.test_loader,
                network_longitudinal=self.network_longitudinal,
                run_dir=self.run_dir,
                save_val=True,
                config=self.config,
            )


if __name__ == "__main__":
    test = Tester()
    test()
