import argparse
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

default_root: str = "/fast_storage/hyeokgi/data_v2_slice_512"
dataset_mode: Literal["base", "longitudinal", "t2pred"] = "longitudinal"
default_run_dir: str = "/home/intern2/long3d/code_recon/log"

DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)
DATASET_MODE: Literal["base", "longitudinal", "t2pred"] = os.environ.get(
    "DATASET_MODE", dataset_mode
)
TRAIN_ITER: int = int(os.environ.get("TRAIN_ITER", 1))  # noqa: PLW1508
RUN_DIR: str = os.environ.get("RUN_DIR", default_run_dir)

if DATASET_MODE == "base":
    ### Base Dataset
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/data_flairt1t2_reg_fastmri_slice/train",
        f"{DATA_ROOT}/data_flairt1t2_reg_fastmri_slice/train",
        f"{DATA_ROOT}/data_flairt1t2_reg_oasis3_slice/train",
        f"{DATA_ROOT}/data_flairt1t2_reg_snuhospital_slice/train",
    ] * TRAIN_ITER

    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/data_flairt1t2_reg_fastmri_slice/val",
        f"{DATA_ROOT}/data_flairt1t2_reg_oasis3_slice/val",
        f"{DATA_ROOT}/data_flairt1t2_reg_snuhospital_slice/val",
    ]

    TEST_DATASET: list[str] = [
        f"{DATA_ROOT}/data_flairt1t2_reg_fastmri_slice/test",
        f"{DATA_ROOT}/data_flairt1t2_reg_oasis3_slice/test",
        f"{DATA_ROOT}/data_flairt1t2_reg_snuhospital_slice/test",
    ]
elif DATASET_MODE == "longitudinal":
    ### Longitudinal Dataset
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/train",
    ] * TRAIN_ITER

    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/valid",
    ]

    TEST_DATASET: list[str] = [
        f"{DATA_ROOT}/test",
    ]
elif DATASET_MODE == "t2pred":
    ### T2 Prediction Dataset
    TRAIN_DATASET: list[str] = [
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/train_fastmri",
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/train_fastmri",
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/train_oasis3",
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/train_snuhospital",
    ] * TRAIN_ITER
    VALID_DATASET: list[str] = [
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/val_fastmri",
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/val_oasis3",
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/val_snuhospital",
    ]
    TEST_DATASET: list[str] = [
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/test_fastmri",
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/test_oasis3",
        f"{DATA_ROOT}/data_predict/data_t2predflair_slice/test_snuhospital",
    ]
else:
    raise ValueError(
        f"Invalid DATASET_MODE: {DATASET_MODE}. Choose 'base' or 'longitudinal' or 't2pred'."
    )
    sys.exit(1)


@dataclass
class GeneralConfig:
    # Dataset
    train_dataset: list[str] = field(default_factory=lambda: TRAIN_DATASET)
    valid_dataset: list[str] = field(default_factory=lambda: VALID_DATASET)
    test_dataset: list[str] = field(default_factory=lambda: TEST_DATASET)
    data_type: str = "*.mat"
    debugmode: bool = False
    prev_checkpoint: str = ""

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = Path(RUN_DIR)
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["longrecon"] = "longrecon"

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adam"
    loss_model: Literal["l1", "l2"] = "l2"
    lr: float = 1e-4
    lr_decay: float = 0.94
    lr_tol: int = 2

    # Train params
    gpu: str = "1"
    train_batch: int = 16
    valid_batch: int = 1
    train_epoch: int = 100
    logging_density: int = 4
    valid_interval: int = 2
    valid_tol: int = 2
    num_workers: int = 16
    save_val: bool = True
    parallel: bool = False
    device: torch.device | None = None
    save_max_idx: int = 500

    # hyper
    use_meta: bool = False
    rotation_conf: str = "rand_10"  # rand_n, fix_n, none_0
    prior_key: str = "out"
    target_key: str = "fl"
    target_mask_key: str = "fl_mask"

    tag: str = ""


@dataclass
class ModelConfig:
    # ReconNet architecture
    recon_net_chan: int = 48
    recon_net_pool: int = 5
    block_type: Literal["block1", "block2", "block3"] = "block2"
    input_depth: int = 16
    input_num: int = 3
    meta_dim: int = 5
    acs_num: int = 24
    parallel_factor: int = 8

    # MaskNet
    masknet_chan: int = 32
    masknet_conv1d_kernel: int = 3
    masknet_num_first_layers: int = 16

    # Sudorecon
    sudorecon_net_chan: int = 32
    sudorecon_net_pool: int = 4

    # Registration
    regnet_chan: int = 24
    regnet_pool: int = 5

    # Other params
    sampling_scheme: Literal["fixed", "adaptive"] = "fixed"  # IMPORTANT: was "adaptive" before
    using_prior: bool = True
    using_registration: bool = True
    using_consistency: bool = True
    variational_k: bool = False


@dataclass
class TestConfig:
    # Dataset
    longitudinal_checkpoints: str = ""


# Argparser
parser = argparse.ArgumentParser(description="Training Configuration")
general_config_dict = asdict(GeneralConfig())
model_config_dict = asdict(ModelConfig())
test_config_dict = asdict(TestConfig())

for key, default_value in general_config_dict.items():
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

for key, default_value in model_config_dict.items():
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

for key, default_value in test_config_dict.items():
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

# Apply argparser
config = GeneralConfig()
modelconfig = ModelConfig()
args = parser.parse_args()

for key, value in vars(args).items():
    if value is not None:
        if hasattr(config, key):
            if isinstance(getattr(config, key), bool):
                setattr(config, key, bool(value))
            else:
                setattr(config, key, value)

        if hasattr(modelconfig, key):
            if isinstance(getattr(modelconfig, key), bool):
                setattr(modelconfig, key, bool(value))
            else:
                setattr(modelconfig, key, value)
