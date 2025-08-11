import torch
from torch import Tensor, nn
from torch.nn import functional


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chans),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.layer1(x)
        return x


class FirstLayer(nn.Module):
    def __init__(
        self,
        chans: int,
        in_chans: int,
    ) -> None:
        super().__init__()

        self.img1 = ConvBlock(
            in_chans=in_chans,
            out_chans=chans // 2,
        )
        self.img2 = ConvBlock(
            in_chans=1,
            out_chans=chans // 2,
        )

        self.final_conv = ConvBlock(
            in_chans=chans,
            out_chans=chans,
        )

    def forward(
        self,
        x: tuple[Tensor, Tensor],
    ) -> Tensor:
        img1, img2 = x
        img1_out = self.img1(img1)
        img2_out = self.img2(img2)
        output = torch.cat([img1_out, img2_out], dim=1)
        output = self.final_conv(output)
        return output


class RegistrationNet(nn.Module):
    def __init__(
        self,
        chans: int = 32,
        num_pool_layers: int = 5,
        init_img_size: int = 128,
        in_chans: int = 1,
    ):
        super().__init__()

        self.num_pool_layers = num_pool_layers
        self.init_img_size = init_img_size

        self.first_layer = FirstLayer(chans, in_chans)

        # Down-sampling layers
        self.down_sample_layers = self.create_down_sample_layers(
            chans=chans,
            num_pool_layers=num_pool_layers,
        )

        # Bottleneck layer
        self.bottleneck_conv = ConvBlock(
            in_chans=chans * (2 ** (num_pool_layers - 1)),
            out_chans=chans * (2 ** (num_pool_layers - 1)),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(chans * (2 ** (num_pool_layers - 1)), 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 6),
        )

    def create_down_sample_layers(
        self,
        chans: int,
        num_pool_layers: int,
    ):
        layers = nn.ModuleList([])
        ch = chans
        for _ in range(num_pool_layers - 1):
            layers.append(
                ConvBlock(
                    ch,
                    ch * 2,
                )
            )
            ch *= 2
        return layers

    def forward(
        self,
        x: tuple[Tensor, Tensor],
    ) -> Tensor:
        if not isinstance(x, tuple) or len(x) != 2:
            raise ValueError("Input must be a tuple of two tensors (img1, img2).")

        x1, x2 = x

        x1 = torch.nn.functional.interpolate(
            input=x1,
            size=(self.init_img_size, self.init_img_size, self.init_img_size),
            mode="trilinear",
        )
        x2 = torch.nn.functional.interpolate(
            input=x2,
            size=(self.init_img_size, self.init_img_size, self.init_img_size),
            mode="trilinear",
        )

        x = (x1, x2)

        output = self.first_layer(x)

        for layer in self.down_sample_layers:
            output = layer(output)
            output = functional.max_pool3d(output, kernel_size=2)

        output = self.bottleneck_conv(output)
        output = functional.adaptive_avg_pool3d(output, (1, 1, 1)).view(output.size(0), -1)
        output = self.output_mlp(output)
        
        return output

if __name__=="__main__":
    # Example usage
    model = RegistrationNet()
    img1 = torch.randn(2, 1, 16, 256, 256)  # Example input tensor
    img2 = torch.randn(2, 1, 16, 256, 256)  # Example input tensor
    output = model((img1, img2))
    print(output.shape)  # Should print torch.Size([1, 6])