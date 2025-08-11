from typing import Literal

import torch
from model.backbone.tunet import VariationalTimeUnet
from model.registration_net import RegistrationNet
from model.sampling_net import SamplingNet
from model.sudorecon_net import SudoreconNet
from torch import Tensor, nn

from common.logger import logger
from common.utils import (
    validate_tensor_channels,
    validate_tensor_dimensions,
    validate_tensors,
)
from components.registration import rotate_and_translate_with_scipy, rotate_and_translate_3d_with_scipy
from components.sampling import apply_adaptive_mask, apply_fixed_mask
from params import ModelConfig


def tensor_5d_2_complex(
    img: Tensor,
) -> Tensor:
    validate_tensor_dimensions([img], 5)  # [B, Z, H, W, C]
    if img.shape[-1] != 2:
        raise ValueError(f"Expected last dimension to be 2, got {img.shape[-1]} instead.")

    img_complex = img[..., 0] + 1j * img[..., 1]
    return img_complex.type(torch.complex64)


def tensor_complex_2_5d(
    img: Tensor,
) -> Tensor:
    validate_tensor_dimensions([img], 4)  # [B, Z, H, W]

    img_real = img.real.unsqueeze(-1)
    img_imag = img.imag.unsqueeze(-1)
    img_5d = torch.cat([img_real, img_imag], dim=-1)
    return img_5d.type(torch.float32)


class LongRecon(torch.nn.Module):
    # Model parameters
    num_timesteps: int
    device: torch.device
    sampling_scheme: Literal["fixed", "adaptive"]

    # Sampling params
    acs_num: int
    parallel_factor: int

    # Flags
    using_prior: bool
    using_registration: bool
    using_consistency: bool
    variational_k: bool

    # Fixed mask params
    fixed_thresh: nn.Parameter
    mask_fix: bool

    # Input params
    input_num: int
    input_depth: int

    # Networks
    recon_net: VariationalTimeUnet
    """
    recon_net: VariationalTimeUnet\n
    Input: tuple[Tensor[B, 2, Z, H, W], Tensor[B, 2, Z, H, W], Tensor[B, 1, Z, H, W]]\n
    Output: Tensor[B, 2, Z, H, W]
    """
    sampling_net: SamplingNet | None
    sudorecon_net: SudoreconNet | None
    """
    sudorecon_net: SudoreconNet\n
    input: Tensor[B, 1, Z, H, W]\n
    output: Tensor[B, 1, Z, H, W]
    """
    registration_net: RegistrationNet | None
    """
    registration_net: RegistrationNet\n
    input: tuple[Tensor[B, 1, Z, H, W], Tensor[B, 1, Z, H, W]]\n
    output: Tensor[B, 6]
    """

    def __init__(
        self,
        device: torch.device,
        modelconfig: ModelConfig,
    ) -> None:
        super().__init__()
        # Model parameters
        self.num_timesteps: int = 1000
        self.device: torch.device = device
        self.sampling_scheme: Literal["fixed", "adaptive"] = modelconfig.sampling_scheme

        # Sampling params
        self.acs_num: int = modelconfig.acs_num
        self.parallel_factor: int = modelconfig.parallel_factor

        # Flags
        self.using_prior: bool = modelconfig.using_prior
        self.using_registration: bool = modelconfig.using_registration
        self.using_consistency: bool = modelconfig.using_consistency
        self.variational_k: bool = modelconfig.variational_k

        # Fixed mask params
        self.fixed_thresh: nn.Parameter = nn.Parameter(
            torch.rand(1, 1000) * 0.9, requires_grad=False
        )
        if self.sampling_scheme == "adaptive":
            self.mask_fix: bool = False
        else:
            self.mask_fix: bool = True

        # Input params
        self.input_num: int = modelconfig.input_num
        self.input_depth: int = modelconfig.input_depth

        # Networks
        self.recon_net = VariationalTimeUnet(
            input_number=modelconfig.input_num,
            input_depth=1,
            out_chans=2,
            meta_dim=modelconfig.meta_dim,
            chans=modelconfig.recon_net_chan,
            num_pool_layers=modelconfig.recon_net_pool,
            time_emb_dim=256,
            block_type=modelconfig.block_type,
        )

        if self.sampling_scheme in ["adaptive"]:
            self.sampling_net = SamplingNet(
                chans=modelconfig.masknet_chan,
                conv1d_kernel=modelconfig.masknet_conv1d_kernel,
                masknet_num_first_layers=modelconfig.masknet_num_first_layers,
            )
        else:
            self.sampling_net = None

        if self.using_registration:
            self.sudorecon_net = SudoreconNet(
                chans=modelconfig.sudorecon_net_chan,
                pools=modelconfig.sudorecon_net_pool,
                in_chans=1,
                out_chans=1,
            )
            self.registration_net = RegistrationNet(
                chans=modelconfig.regnet_chan,
                num_pool_layers=modelconfig.regnet_pool,
            )
        else:
            self.sudorecon_net = None
            self.registration_net = None

    def _run_recon_net(
        self,
        input: tuple[Tensor, Tensor, Tensor],
        t: Tensor,
        m: Tensor,
        label: Tensor,
        mask: Tensor,
        consistency: bool,
    ) -> Tensor:
        for _input in input:
            validate_tensors([_input])

        pred = self.recon_net(
            x=input,
            t=t.type(torch.float32),
            m=m.type(torch.float32),
            label=label,
            mask=mask.type(torch.float32),
            consistency=consistency,
        )
        return pred

    def _undersample_img(
        self,
        img: Tensor,
        quantize: bool,
        variational_k: bool,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
    ]:
        validate_tensors([img])
        validate_tensor_dimensions([img], 5)  # [B, Z, H, W, C]
        validate_tensor_channels(img, self.input_depth)
        B, _, H, _, _ = img.shape
        img_complex = tensor_5d_2_complex(img)

        if self.sampling_scheme in ["fixed"]:
            img_undersample, mask, mask_prob = apply_fixed_mask(
                img=img_complex,
                acs_num=self.acs_num,
                parallel_factor=self.parallel_factor,
            )
        elif self.sampling_scheme in ["adaptive"]:
            if self.mask_fix:
                with torch.no_grad():
                    sampling_output = self.sampling_net.forward(H)
            else:
                sampling_output = self.sampling_net.forward(H)

            # to batch
            sampling_output = sampling_output.repeat(B, 1)

            img_undersample, mask, mask_prob = apply_adaptive_mask(
                img=img_complex,
                sampling_output=sampling_output,
                acs_num=self.acs_num,
                parallel_factor=self.parallel_factor,
                quantize=quantize,
                fixed_thresh=self.fixed_thresh,
                mask_fix=self.mask_fix,
                variational_k=variational_k,
            )
        else:
            raise NotImplementedError

        img_undersample = tensor_complex_2_5d(img_undersample)
        validate_tensor_dimensions([img_undersample], 5)  # [B, Z, H, W, C]

        return (
            img_undersample,
            mask.clone().detach(),
            mask_prob,
        )

    def set_fixed_thresh(
        self,
    ) -> None:
        if self.sampling_scheme in ["adaptive"]:
            self.mask_fix = True

    def forward(
        self,
        prior: Tensor, # shape: [B, Z, H, W]
        prior_rot: Tensor, # shape: [B, Z, H, W]
        target: Tensor, # shape: [B, Z, H, W, 2]
        time: Tensor, # shape: [B, 1]
        meta: Tensor, # shape: [B, N]
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        """
        Args:
            prior (Tensor): Prior image tensor of shape [B, Z, H, W].
            prior_rot (Tensor): Rotated prior image tensor of shape [B, Z, H, W].
            target (Tensor): Target image tensor of shape [B, Z, H, W, C].
            time (Tensor): Time tensor of shape [B, 1].
            meta (Tensor): Meta information tensor [B, N].
        """
        validate_tensors([prior, prior_rot, target, time])
        validate_tensor_dimensions([prior, prior_rot], 4)  # [B, Z, H, W]
        validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C=2]
        for _img in [prior, prior_rot, target]:
            validate_tensor_channels(_img, self.input_depth)

        B, Z, _, _, _ = target.shape

        label = target  # [B, Z, H, W, C=2]
        label = label.permute(0, 4, 1, 2, 3)  # [B, C=2, Z, H, W]
        noise = torch.randn_like(label).to(self.device) # [B, C=2, Z, H, W]

        t = time.squeeze(1).long()
        t_n = t / self.num_timesteps

        x_t = ((1 - t_n).view(B, 1, 1, 1) * label + t_n.view(B, 1, 1, 1) * noise).type(
            torch.float32
        ) # [B, C=2, Z, H, W]

        (
            label_undersample, # [B, Z, H, W, C=2]
            mask, # [B, 1, H, W]
            mask_prob,
        ) = self._undersample_img(
            img=target, # [B, Z, H, W, C=2]
            quantize=False,
            variational_k=self.variational_k,
        )
        
        # uniform z mask
        mask = mask.unsqueeze(1).expand(-1, -1, Z, -1, -1)

        if self.using_registration:
            label_undersample_complex = tensor_5d_2_complex(label_undersample).clone().detach() # [B, Z, H, W], complex64
            sudo_recon_input = torch.abs(label_undersample_complex).unsqueeze(1) # [B, Z, H, W]->[B, C=1, Z, H, W], float32
            sudo_recon = self.sudorecon_net.forward(x=sudo_recon_input) # [B, C=1, Z, H, W]
            reg_pred = self.registration_net(x=(prior_rot.unsqueeze(1), sudo_recon.clone().detach()))
            # added unsqeeze(1) to prior_rot for empty channel dimension
        else:
            sudo_recon = None
            reg_pred = None

        prior_cond = prior.unsqueeze(1) if self.using_prior else torch.zeros_like(prior) # [B, 1, Z, H, W]
        label_cond = label_undersample.permute(0, 4, 1, 2, 3)  # [B, C, Z, H, W]
        print("x_t shape:", x_t.shape, "should be [B, C=2, Z, H, W]")
        print("prior_cond shape:", prior_cond.shape, "should be [B, 1, Z, H, W]")
        print("label_cond shape:", label_cond.shape, "should be [B, C=2, Z, H, W]")
        pred = self._run_recon_net(
            input=[
                x_t, # noisy label [B, C, H, W] -> [B, 2, Z, H, W]
                label_cond, # undersampled label [B, C, H, W] -> [B, 2, Z, H, W]
                prior_cond, # prior image, depth 3 [B, Z, H, W] -> [B, 1, Z, H, W]
            ],
            t=t_n, # time normalized [B, 1]
            m=meta, # meta information [B, N]
            label=label, # full label [B, C=2, Z, H, W]
            mask=mask, # mask [B, C=1, Z, H, W]
            consistency=False,
        )
        print("forward end")
        return (
            pred, # shape: [B, C=2, Z, H, W]
            label, # shape: [B, C=2, Z, H, W]
            sudo_recon, # shape: [B, C=1, Z, H, W]
            reg_pred, # shape: [B, 6] or None
            mask_prob, # shape: [B, 1, H, W] or None
        )

    @torch.inference_mode()
    def undersample_img(
        self,
        img: Tensor, # [B, Z, H, W, C=2]
        quantize: bool,
        variational_k: bool,
    ) -> tuple[
        Tensor, # [B, Z, H, W, C=2]
        Tensor, # [B, 1, H, W]
        Tensor | None,
    ]:
        validate_tensors([img])
        validate_tensor_dimensions([img], 5)  # [B, Z, H, W, C]
        validate_tensor_channels(img, self.input_depth)

        img_undersample, mask, mask_prob = self._undersample_img(
            img=img, quantize=quantize, variational_k=variational_k
        )

        validate_tensor_dimensions([img_undersample], 5)  # [B, Z, H, W, C]

        return (
            img_undersample,
            mask,
            mask_prob,
        )

    @torch.inference_mode()
    def sudo_recon_img(
        self,
        img_undersample: Tensor,
    ) -> Tensor:
        validate_tensors([img_undersample])
        validate_tensor_dimensions([img_undersample], 4)
        validate_tensor_channels(img_undersample, 1)

        return self.sudorecon_net.forward(x=img_undersample)

    @torch.inference_mode()
    def registration_prior_to_target(
        self,
        prior: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        validate_tensors([prior, target])
        validate_tensor_dimensions([prior, target], 4)
        validate_tensor_channels(prior, self.input_depth)
        validate_tensor_channels(target, 1)

        reg_pred = self.registration_net(x=(prior, target))

        # deprecated code, change to 3d rotation
        # prior_reg = rotate_and_translate_with_scipy(
        #     img=prior,
        #     angle=-reg_pred[0, 0].cpu(),
        #     tx=-reg_pred[0, 1].cpu(),
        #     ty=-reg_pred[0, 2].cpu(),
        # ).to(prior.device)

        prior_reg = rotate_and_translate_3d_with_scipy(
            img=prior,
            angle_x=-reg_pred[0, 0].cpu(),
            angle_y=-reg_pred[0, 1].cpu(),
            angle_z=-reg_pred[0, 2].cpu(),
            tx=-reg_pred[0, 3].cpu(),
            ty=-reg_pred[0, 4].cpu(),
            tz=-reg_pred[0, 5].cpu(),
        ).to(prior.device)

        return prior_reg, reg_pred

    @torch.inference_mode()
    def flow_reverse(
        self,
        label_cond: Tensor,
        prior_cond: Tensor,
        meta: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Args:
            label_cond (Tensor): Condition tensor of shape [B, C, H, W].
            prior_cond (Tensor): Prior condition tensor of shape [B, Z, H, W].
            meta (Tensor): Meta information tensor [B, N].
            mask (Tensor): Mask tensor of shape [B, 1, H, W].
        """
        validate_tensors([label_cond, prior_cond, mask])
        validate_tensor_dimensions([label_cond, prior_cond, mask], 4)
        validate_tensor_channels(prior_cond, self.input_depth)
        validate_tensor_channels(label_cond, 2)

        B, _, _, _ = label_cond.shape

        times = torch.tensor([1000, 800, 600, 400, 300, 200, 150, 100, 50, 0], dtype=torch.long)
        dts = (times[:-1] - times[1:]) / self.num_timesteps
        times = times[:-1]

        t = torch.tensor(self.num_timesteps, device=self.device, dtype=torch.long)
        noise = torch.randn_like(label_cond).to(self.device)

        t_n = t / self.num_timesteps
        t_n = t_n.repeat(B).view(B, 1, 1, 1)
        x_t = (t_n * noise).type(torch.float32)

        for dt, time in zip(dts, times, strict=True):
            logger.trace(f"Diffusion time : {time}")
            t_batch = (
                torch.full(
                    size=(B,),
                    fill_value=time.item() if isinstance(time, torch.Tensor) else time,
                    device=self.device,
                    dtype=torch.long,
                )
                / self.num_timesteps
            )

            pred = self._run_recon_net(
                input=[
                    x_t,
                    label_cond,
                    prior_cond,
                ],
                t=t_batch,
                m=meta,
                label=label_cond,
                mask=mask,
                consistency=self.using_consistency,
            )

            network_output = noise - pred
            network_output = network_output.type_as(x_t)

            validate_tensors([network_output])

            x_next = x_t - dt * network_output
            x_t = x_next

        return x_t.type(torch.float32)

    @torch.inference_mode()
    def long_recon(
        self,
        prior: Tensor,
        prior_rot: Tensor,
        target: Tensor,
        meta: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
    ]:
        """
        Args:
            prior (Tensor): Prior image tensor of shape [B, Z, H, W, C].
            prior_rot (Tensor): Rotated prior image tensor of shape [B, Z, H, W].
            target (Tensor): Target image tensor of shape [B, Z, H, W, C].
            meta (Tensor): Meta information tensor [B ,N].
        """
        validate_tensors([prior, target])
        validate_tensor_dimensions([prior], 4)  # [B, Z, H, W]
        validate_tensor_channels(prior, self.input_depth)
        validate_tensor_dimensions([target], 5)  # [B, Z, H, W, C]
        validate_tensor_channels(target, self.input_depth)

        (
            target_undersample,
            mask,
            mask_prob,
        ) = self.undersample_img(
            img=target,
            quantize=True,
            variational_k=False,
        )

        prior = prior_rot if self.using_prior else torch.zeros_like(prior_rot)
        z_middle = target.shape[1] // 2

        label_undersample = target_undersample[
            :, z_middle : z_middle + 1, :, :, :
        ]  # [B, 1, H, W, C]
        label_cond = label_undersample.transpose(1, 4).squeeze(4)  # [B, C, H, W]

        if self.using_registration:
            label_undersample_complex = tensor_5d_2_complex(label_undersample).clone().detach()
            sudo_recon_input = torch.abs(label_undersample_complex)
            sudo_recon = self.sudo_recon_img(img_undersample=sudo_recon_input)
            prior_reg, _ = self.registration_prior_to_target(prior=prior_rot, target=sudo_recon)
            prior_cond = prior_reg
        else:
            sudo_recon = None
            prior_reg = None
            prior_cond = prior

        x_t = self.flow_reverse(
            label_cond=label_cond,
            prior_cond=prior_cond,
            meta=meta,
            mask=mask,
        )

        return (
            x_t.type(torch.float32),
            label_undersample.type(torch.float32),
            mask,
            mask_prob,
            sudo_recon,
            prior_reg,
        )
