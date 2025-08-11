import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional


from common.utils import validate_tensors


# temporary function to validate tensors, uncomment when needed
# def validate_tensors(tensors: list[Tensor]) -> None:
#     for i, t in enumerate(tensors):
#         if not isinstance(t, Tensor):
#             raise TypeError(f"Tensor at index {i} is not a torch.Tensor, got {type(t)} instead.")


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(100) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        emb = self.linear(emb)

        return emb


class TimeConvAttentionBlock(nn.Module):
    def __init__(
        self,
        input_chans: int,
        out_chans: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(input_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_chans * 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

        self.meta_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, out_chans * 2),
        )

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        m_emb: Tensor,
    ) -> Tensor:
        x = self.layer1(x)

        t_emb = self.time_mlp(t_emb)[:, :, None, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(x)

        m_emb = self.meta_mlp(m_emb)[:, :, None, None, None]
        shift, bias = m_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        return x


def get_time_conv_attention_block(
    block_type: str,
    input_chans: int,
    out_chans: int,
    time_emb_dim: int,
) -> nn.Module:
    if block_type == "block1" or block_type == "block2" or block_type == "block3":
        # print(f"Input channels: {input_chans}, Output channels: {out_chans}, Time embedding dim: {time_emb_dim}")
        return TimeConvAttentionBlock(
            input_chans=input_chans,
            out_chans=out_chans,
            time_emb_dim=time_emb_dim,
        )
    else:
        raise NotImplementedError(
            f"Block type '{block_type}' is not implemented yet. " "Available types: 'block1~3'."
        )


class FirstLayer(nn.Module):
    def __init__(
        self,
        input_number: int,
        input_depth: int,
        out_chans: int,
        chans: int,
        time_emb_dim: int,
        block_type: Literal["block1", "block2", "block3"] = "block1",
    ) -> None:
        super().__init__()
        self.input_number = input_number

        chan_list: list[int] = [out_chans, out_chans, input_depth]

        self.blocks = nn.ModuleList(
            [
                get_time_conv_attention_block(
                    block_type=block_type,
                    input_chans=chan_list[i],
                    out_chans=chans // 2,
                    time_emb_dim=time_emb_dim,
                )
                for i in range(input_number)
            ]
        )

        self.compose = get_time_conv_attention_block(
            block_type=block_type,
            input_chans=(chans // 2) * input_number,
            out_chans=(chans // 2) * input_number,
            time_emb_dim=time_emb_dim,
        )

    def forward(
        self,
        x: tuple[Tensor, ...],
        t_emb: Tensor,
        m_emb: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert (
            len(x) == self.input_number
        ), f"Expected {self.input_number} input tensors, got {len(x)}"

        for tensor in x:
            validate_tensors([tensor])

        outputs = [block(inp, t_emb, m_emb) for block, inp in zip(self.blocks, x, strict=True)]
        x_cat = torch.cat(x, dim=1)
        x_feat_cat = torch.cat(outputs, dim=1)
        x_out = self.compose(x_feat_cat, t_emb, m_emb)

        return x_cat, x_feat_cat, x_out


class FinalLayer(nn.Module):
    def __init__(
        self,
        x_chans: int,
        x_feat_chans: int,
        feature_chans: int,
        out_chans: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(feature_chans + x_feat_chans, feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_chans),
            nn.SiLU(inplace=True),
        )

        self.meta_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, feature_chans * 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(feature_chans + x_chans, feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_chans),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, feature_chans * 2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_chans),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_chans, feature_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_chans),
            nn.SiLU(inplace=True),
            nn.Conv3d(feature_chans, out_chans, kernel_size=1, padding=0),
        )

    def forward(
        self,
        x_cat: Tensor,
        x_feat_cat: Tensor,
        output: Tensor,
        t_emb: Tensor,
        m_emb: Tensor,
    ) -> Tensor:

        x = self.layer1(torch.cat([output, x_feat_cat], dim=1))

        m_emb = self.meta_mlp(m_emb)[:, :, None, None, None]
        shift, bias = m_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(torch.cat([x, x_cat], dim=1))

        t_emb = self.time_mlp(t_emb)[:, :, None, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer3(x)

        return x


class TimeUnet(nn.Module):
    def __init__(
        self,
        input_number: int = 3,
        input_depth: int = 3,
        out_chans: int = 1,
        meta_dim: int = 7,
        chans: int = 64,
        num_pool_layers: int = 6,
        time_emb_dim: int = 256,
        block_type: Literal["block1", "block2", "block3"] = "block2",
    ):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_emb_dim)

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
    
        self.first_layer = FirstLayer(
            input_number=input_number,
            input_depth=input_depth,
            out_chans=out_chans,
            chans=chans,
            time_emb_dim=time_emb_dim,
            block_type=block_type,
        )

        chans = (chans // 2) * input_number

        self.down_pool_layers = self.create_down_pool_layers(
            chans=chans,
            num_pool_layers=num_pool_layers,
        )

        self.down_layers = self.create_down_layers(
            chans=chans,
            num_pool_layers=num_pool_layers,
            time_emb_dim=time_emb_dim,
            block_type=block_type,
        )

        self.bottleneck_conv = get_time_conv_attention_block(
            block_type=block_type,
            input_chans=chans * (2 ** (num_pool_layers - 1)),
            out_chans=chans * (2 ** (num_pool_layers - 1)),
            time_emb_dim=time_emb_dim,
        )

        self.up_conv_layers = self.create_up_conv_layers(
            chans=chans,
            num_pool_layers=num_pool_layers,
        )

        self.up_layers = self.create_up_layers(
            chans=chans,
            num_pool_layers=num_pool_layers,
            time_emb_dim=time_emb_dim,
            block_type=block_type,
        )

        self.final_conv = FinalLayer(
            x_chans=input_depth * (input_number - 2) + out_chans * 2,
            x_feat_chans=chans,
            feature_chans=chans,
            out_chans=out_chans,
            time_emb_dim=time_emb_dim,
        )

    def create_down_pool_layers(
        self,
        chans: int,
        num_pool_layers: int,
    ):
        layers = nn.ModuleList()
        ch = chans
        for _ in range(num_pool_layers - 1):
            layers.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1))
            ch *= 2
        return layers

    def create_down_layers(
        self,
        chans: int,
        num_pool_layers: int,
        time_emb_dim: bool,
        block_type: Literal["block1", "block2", "block3"] = "block1",
    ):
        layers = nn.ModuleList([])
        ch = chans
        layers.append(
            get_time_conv_attention_block(
                block_type=block_type,
                input_chans=ch,
                out_chans=ch * 2,
                time_emb_dim=time_emb_dim,
            )
        )
        ch *= 2
        for _ in range(num_pool_layers - 2):
            layers.append(
                get_time_conv_attention_block(
                    block_type=block_type,
                    input_chans=ch,
                    out_chans=ch * 2,
                    time_emb_dim=time_emb_dim,
                )
            )
            ch *= 2
        return layers

    def create_up_conv_layers(
        self,
        chans: int,
        num_pool_layers: int,
    ):
        layers = nn.ModuleList()
        ch = chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            layers.append(nn.ConvTranspose3d(ch, ch, kernel_size=2, stride=2))
            ch //= 2
        layers.append(nn.Identity())
        return layers

    def create_up_layers(
        self,
        chans: int,
        num_pool_layers: int,
        time_emb_dim: bool,
        block_type: Literal["block1", "block2", "block3"] = "block1",
    ):
        layers = nn.ModuleList()
        ch = chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            layers.append(
                get_time_conv_attention_block(
                    block_type=block_type,
                    input_chans=ch * 2,
                    out_chans=ch // 2,
                    time_emb_dim=time_emb_dim,
                )
            )
            ch //= 2
        layers.append(
            get_time_conv_attention_block(
                block_type=block_type,
                input_chans=ch * 2,
                out_chans=ch,
                time_emb_dim=time_emb_dim,
            )
        )
        return layers

    def forward(
        self,
        x: tuple[Tensor, ...],
        t: Tensor,
        m: Tensor,
    ) -> Tensor:
        t_emb = self.time_mlp(t)
        m_emb = self.meta_mlp(m)

        stack: list[Tensor] = []
        x_cat, x_feat_cat, output = self.first_layer(x, t_emb, m_emb)
        stack.append(output)

        # print(
        #     f"Input shape: {x_cat.shape}, Feature shape: {x_feat_cat.shape}, Output shape: {output.shape}"
        # )

        for down_pool, layer in zip(self.down_pool_layers, self.down_layers, strict=False):
            output = layer(output, t_emb, m_emb)
            # print(f"Down layer output shape: {output.shape}")
            stack.append(output)
            output = down_pool(output)

        output = self.bottleneck_conv(output, t_emb, m_emb)
        # print(f"Bottleneck output shape: {output.shape}")

        for up_conv, layer in zip(self.up_conv_layers, self.up_layers, strict=False):
            downsampled_output = stack.pop()
            output = up_conv(output)
            # print(f"Up conv output shape: {output.shape}, Downsampled output shape: {downsampled_output.shape}")

            B, C, D, W, H = downsampled_output.shape
            output = output[:, :, :D, :W, :H]

            output = torch.cat([output, downsampled_output], dim=1)
            output = layer(output, t_emb, m_emb)
        # print(f"Final output shape before final conv: {output.shape}")

        output = self.final_conv(x_cat, x_feat_cat, output, t_emb, m_emb)

        return output


def ifft2c(img_k: torch.Tensor) -> torch.Tensor:
    img = torch.fft.ifftn(torch.fft.ifftshift(img_k, dim=(-2, -1)), dim=(-2, -1))
    return img


def fft2c(img: torch.Tensor) -> torch.Tensor:
    img_k = torch.fft.fftshift(torch.fft.fftn(img, dim=(-2, -1)), dim=(-2, -1))
    return img_k


def channl_2_complex(img: Tensor) -> Tensor:
    return (img[:, 0, :, :] + 1j * img[:, 1, :, :]).unsqueeze(1)


def complex_2_channel(img: Tensor) -> Tensor:
    return torch.stack([img.real, img.imag], dim=1).squeeze(2)


class VariationalTimeUnet(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.timeunet = TimeUnet(
            *args,
            **kwargs,
        )

    def forward(
        self,
        label: Tensor,
        x: tuple[Tensor, ...],
        t: Tensor,
        m: Tensor,
        mask: Tensor,
        consistency: bool,
    ) -> Tensor:
        x0_pred = self.timeunet.forward(
            x=x,
            t=t,
            m=m,
        )

        if consistency:
            label_complex = channl_2_complex(label)
            x0_pred_complex = channl_2_complex(x0_pred)
            con_k = fft2c(label_complex - x0_pred_complex) * mask

            x0_pred_kspace_complex = fft2c(x0_pred_complex) + con_k

            x0_pred_complex = ifft2c(x0_pred_kspace_complex)
            x0_pred = complex_2_channel(x0_pred_complex)
            return x0_pred
        else:
            return x0_pred


if __name__ == "__main__":
    # Example usage
    model = VariationalTimeUnet(
        input_number=3,
        input_depth=1,
        out_chans=2,
        meta_dim=7,
        chans=64,
        num_pool_layers=3,
        time_emb_dim=256,
        block_type="block2",
    )

    x = (torch.randn(1, 2, 64, 64, 64), torch.randn(1, 2, 64, 64, 64), torch.randn(1, 1, 64, 64, 64))
    t = torch.tensor([0.5])
    m = torch.randn(1, 7)
    mask = torch.ones(1, 1, 64, 64, 64)

    output = model(label=torch.randn(1, 2, 64, 64, 64), x=x, t=t, m=m, mask=mask, consistency=True)
    print(output.shape)