import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from datawrapper.datawrapper import DataWrapper
from model.longitudinal_recon import LongRecon
from scipy.io import savemat
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from common.logger import logger
from common.metric import calculate_psnr, calculate_ssim
from common.utils import (
    seconds_to_dhms,
    validate_tensor_channels,
    validate_tensor_dimensions,
)
from components.metriccontroller import MetricController
from params import ModelConfig, config, modelconfig

NETWORK = LongRecon | torch.nn.DataParallel[LongRecon]
OPTIM = Adam | AdamW


def get_network(
    device: torch.device | None,
    model_type: str,
    modelconfig: ModelConfig,
) -> NETWORK:
    if device is None:
        raise TypeError("device is not to be None")

    if model_type == "longrecon":
        return LongRecon(device=device, modelconfig=modelconfig)
    else:
        raise KeyError("model type not matched")


def get_optim(
    network: NETWORK | None,
    optimizer: str,
) -> OPTIM | None:
    if network is None:
        return None
    if optimizer == "adam":
        return Adam(network.parameters(), betas=(0.9, 0.99))
    elif optimizer == "adamw":
        return AdamW(network.parameters(), betas=(0.9, 0.99), weight_decay=0.0)
    else:
        raise KeyError("optimizer not matched")


def get_loss_func(
    loss_model: str,
) -> Callable:
    if loss_model == "l1":
        return torch.nn.L1Loss(reduction="none")
    elif loss_model == "l2":
        return torch.nn.MSELoss(reduction="none")
    else:
        raise KeyError("loss func not matched")


def get_learning_rate(
    epoch: int,
    lr: float,
    lr_decay: float,
    lr_tol: int,
) -> float:
    factor = epoch - lr_tol if lr_tol < epoch else 0
    return lr * (lr_decay**factor)


def set_optimizer_lr(
    optimizer: OPTIM | None,
    learning_rate: float,
) -> OPTIM | None:
    if optimizer is None:
        return None
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return optimizer


def log_summary(
    init_time: float,
    state: MetricController,
    log_std: bool = False,
) -> None:
    spend_time = seconds_to_dhms(time.time() - init_time)
    for key in state.state_dict:
        if log_std:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e} + {state.std(key):0.3e} "
            logger.info(summary)
        else:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e}"
            logger.info(summary)


def save_checkpoint(
    network: NETWORK,
    run_dir: Path,
    epoch: str | int | None = None,
) -> None:
    if epoch is None:
        epoch = "best"
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "model_config": asdict(modelconfig),
        },
        run_dir / f"checkpoints/checkpoint_{epoch}.ckpt",
    )


def zero_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.zero_grad()


def step_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.step()


def mask_reg(
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.dim() != 4:
        raise NotImplementedError("mask has to be 3D")

    half_len = mask.size(-2) // 2
    mask_half = mask[:, :, :, :half_len]
    diff = mask_half[:, :, :, 1:] - mask_half[:, :, :, :-1]
    penalized_diff = torch.abs(torch.clamp(diff, max=0))
    return torch.mean(penalized_diff, dim=(1, 2, 3), keepdim=True)


def train_epoch_longrecon(
    _data: dict[DataWrapper, Tensor | str],
    network: NETWORK,
    epoch: int,
    train_state: MetricController,
) -> int:
    loss_func = get_loss_func(config.loss_model)

    prior: Tensor = _data[DataWrapper.Prior].to(config.device) # shape: [B, Z, H, W]
    reg_label: Tensor = _data[DataWrapper.RegLabel].to(config.device) # shape: [3], should be [6] for 3D

    img_cnt_minibatch = prior.shape[0] # B

    (
        pred, # shape: [B, 2, H, W] -> [B, 2, Z, H, W]
        label, # shape: [B, 2, H, W] -> [B, 2, Z, H, W]
        sudorecon, # shape: [B, 1, H, W] -> [B, 1, Z, H, W]
        reg_pred, # shape: [B, 3] or [B, 6] for 3D
        mask_prob, # shape: [B, 1, H, W], don't needed
    ) = network.forward(
        prior=prior, # shape: [B, Z, H, W]
        prior_rot=_data[DataWrapper.PriorRot].to(config.device), # perturbated, shape: [B, Z, H, W]
        target=_data[DataWrapper.Target].to(config.device), # shape: [B, Z, H, W, 2]
        time=_data[DataWrapper.Time].to(config.device), # shape: [B, 1]
        meta=_data[DataWrapper.Meta].to(config.device), # shape: [B, N]
    )

    loss_recon = torch.mean(loss_func(pred, label), dim=(1, 2, 3, 4), keepdim=True)

    if (not network.module.mask_fix) and (network.module.sampling_scheme == "adaptive"):
        loss_mask = mask_reg(mask_prob)
        loss_recon += loss_mask

    torch.mean(loss_recon).backward()
    train_state.add("loss_recon", loss_recon)

    if (network.module.mask_fix) and (sudorecon is not None) and (reg_pred is not None):
        label_abs = torch.abs(label[:, 0, :, :] + 1j * label[:, 1, :, :]).unsqueeze(
            1
        )  # [B, 1, H, W]
        loss_sudorecon = torch.mean(loss_func(sudorecon, label_abs), dim=(1, 2, 3, 4), keepdim=True)
        loss_reg = (
            torch.mean(loss_func(reg_pred, reg_label), dim=(1), keepdim=True)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        torch.mean(loss_sudorecon).backward()
        torch.mean(loss_reg).backward()

        train_state.add("loss_sudorecon", loss_sudorecon)
        train_state.add("loss_reg", loss_reg)

    return img_cnt_minibatch


def train_epoch(
    train_loader: DataLoader,
    train_len: int,
    network: NETWORK,
    optim_list: list[OPTIM | None],
    epoch: int,
) -> None:
    train_state = MetricController()
    train_state.reset()
    network.train()

    logging_cnt: int = 1
    img_cnt: int = 0
    for _data in train_loader:
        zero_optimizers(optim_list=optim_list)
        if config.model_type == "longrecon":
            img_cnt_minibatch = train_epoch_longrecon(
                _data=_data,
                network=network,
                epoch=epoch,
                train_state=train_state,
            )
        else:
            raise KeyError("model type not matched")

        step_optimizers(optim_list=optim_list)
        img_cnt += img_cnt_minibatch
        if img_cnt > (train_len / config.logging_density * logging_cnt):
            log_summary(init_time=config.init_time, state=train_state)
            logging_cnt += 1

    log_summary(init_time=config.init_time, state=train_state)


def save_result_to_mat(
    test_dir: Path,
    batch_cnt: int,
    tesner_dict: dict[str, Tensor | None],
    img_cnt: int,
) -> None:
    os.makedirs(test_dir, exist_ok=True)
    save_dict = {}

    if batch_cnt == 0:
        logger.warning("batch_cnt is 0, no data to save")
        return

    for i in range(batch_cnt):
        for key, value in tesner_dict.items():
            if value is not None:
                save_dict[key] = value.cpu().detach().numpy()[i, ...]

        idx = img_cnt + i + 1
        savemat(f"{test_dir}/{idx}_res.mat", save_dict)


def update_metrics(
    test_state: MetricController,
    output_abs: Tensor,
    target_abs: Tensor,
    sudorecon: Tensor | None,
    target_mask: Tensor,
) -> None:
    mask_sum = torch.sum(target_mask, dim=(1, 2, 3), keepdim=True)
    if not torch.any(mask_sum == 0):
        test_state.add("psnr", calculate_psnr(output_abs, target_abs, target_mask))
        test_state.add("ssim", calculate_ssim(output_abs, target_abs, target_mask))
        if sudorecon is not None:
            test_state.add("psnr_sudo", calculate_psnr(sudorecon, target_abs, target_mask))
            test_state.add("ssim_sudo", calculate_ssim(sudorecon, target_abs, target_mask))


def test_part_longrecon(
    _data: dict[DataWrapper, Tensor | str],
    test_dir: Path,
    model: NETWORK,
    save_val: bool,
    test_state: MetricController,
    img_cnt: int,
) -> int:
    loss_func = get_loss_func(config.loss_model)

    prior_rot: Tensor = _data[DataWrapper.PriorRot].to(config.device)
    prior: Tensor = _data[DataWrapper.Prior].to(config.device)
    target: Tensor = _data[DataWrapper.Target].to(config.device)
    target_mask: Tensor = _data[DataWrapper.Mask].to(config.device)
    meta: Tensor = _data[DataWrapper.Meta].to(config.device)

    batch_cnt = prior.shape[0]

    (
        output,
        label_undersample,
        mask,
        mask_prob,
        sudorecon,
        img1reg,
    ) = model.long_recon(
        prior=prior,
        prior_rot=prior_rot,
        target=target,
        meta=meta,
    )

    validate_tensor_dimensions([output], 4)  # [B, C, H, W]
    validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C]

    validate_tensor_channels(prior, model.input_depth)
    c_middle = prior.shape[1] // 2

    output_abs = torch.abs(
        (output[:, 0, :, :] + 1j * output[:, 1, :, :]).unsqueeze(1)
    )  # [B, 1, H, W]
    target_abs = torch.abs(target[:, :, :, :, 0] + 1j * target[:, :, :, :, 1])  # [B, Z, H, W]
    target_abs = target_abs[:, c_middle : c_middle + 1, :, :]  # [B, 1, H, W]

    loss = loss_func(output_abs, target_abs)
    loss = torch.mean(loss, dim=(1, 2, 3), keepdim=True)
    test_state.add("loss_recon", loss)

    update_metrics(
        test_state=test_state,
        output_abs=output_abs,
        target_abs=target_abs,
        sudorecon=sudorecon,
        target_mask=target_mask,
    )

    prior_rot = prior_rot[:, c_middle : c_middle + 1, :, :]
    prior = prior[:, c_middle : c_middle + 1, :, :]
    target = target[:, c_middle : c_middle + 1, :, :, :]

    if save_val:
        save_result_to_mat(
            test_dir=test_dir,
            batch_cnt=batch_cnt,
            tesner_dict={
                "prior": prior,
                "prior_rot": prior_rot,
                "out": output,
                "target": target,
                "label_undersample": label_undersample,
                "mask": mask,
                "meta": meta,
                "mask_prob": mask_prob if mask_prob is not None else None,
                "prior_reg": img1reg if img1reg is not None else None,
                "sudorecon": sudorecon if sudorecon is not None else None,
            },
            img_cnt=img_cnt,
        )

    return batch_cnt


def test_part(
    epoch: int,
    data_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    save_val: bool,
) -> float:
    test_state = MetricController()
    test_state.reset()
    network.eval()
    model = network.module if isinstance(network, torch.nn.DataParallel) else network

    img_cnt: int = 0
    for _data in data_loader:
        if config.model_type in ("longrecon"):
            batch_cnt = test_part_longrecon(
                _data=_data,
                test_dir=run_dir / f"test/ep_{epoch}",
                model=model,
                save_val=save_val and img_cnt <= config.save_max_idx,
                test_state=test_state,
                img_cnt=img_cnt,
            )
        else:
            raise KeyError("model type not matched")

        img_cnt += batch_cnt

    log_summary(init_time=config.init_time, state=test_state, log_std=True)

    primary_metric = test_state.mean("psnr")
    return primary_metric
