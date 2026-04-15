from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath
except Exception:  # pragma: no cover
    try:
        from timm.models.layers import DropPath
    except Exception:  # pragma: no cover
        class DropPath(nn.Identity):  # type: ignore[misc]
            def __init__(self, drop_prob: float = 0.0) -> None:
                super().__init__()


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"unsupported data_format: {data_format}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)

        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * self.weight.view(*shape) + self.bias.view(*shape)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        spatial_dims: int,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if spatial_dims == 3:
            conv = nn.Conv3d(channels, channels, kernel_size=7, padding=3, groups=channels)
        elif spatial_dims == 2:
            conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        else:
            raise ValueError("spatial_dims must be 2 or 3")

        self.spatial_dims = spatial_dims
        self.depthwise = conv
        self.norm = LayerNorm(channels, data_format="channels_last")
        self.fc1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * channels, channels)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        if self.spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.gamma is not None:
            x = self.gamma * x
        if self.spatial_dims == 3:
            x = x.permute(0, 4, 1, 2, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int = 11,
        dims: list[int] | tuple[int, ...] = (16, 32, 64, 128, 128),
        num_res_blocks: list[int] | tuple[int, ...] = (3, 3, 3, 9, 3),
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        if num_frames != 16:
            raise ValueError("this course pipeline expects 16-frame encoder inputs")
        if len(dims) != len(num_res_blocks):
            raise ValueError("dims and num_res_blocks must have the same length")

        self.embed_dim = int(dims[-1])
        self.dims = [int(dim) for dim in dims]
        self.num_res_blocks = [int(blocks) for blocks in num_res_blocks]
        self.num_frames = int(num_frames)

        stem = nn.Sequential(
            nn.Conv3d(in_chans, self.dims[0], kernel_size=(1, 4, 4), padding="same"),
            LayerNorm(self.dims[0], data_format="channels_first"),
        )
        self.downsample_layers = nn.ModuleList([stem])

        for idx in range(len(self.dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(self.dims[idx], data_format="channels_first"),
                    nn.Conv3d(self.dims[idx], self.dims[idx + 1], kernel_size=2, stride=2),
                )
            )

        self.res_blocks = nn.ModuleList()
        last_stage = len(self.dims) - 1
        for idx, channels in enumerate(self.dims):
            spatial_dims = 2 if idx == last_stage else 3
            self.res_blocks.append(
                nn.Sequential(
                    *[
                        ResidualBlock(channels, spatial_dims=spatial_dims)
                        for _ in range(self.num_res_blocks[idx])
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for downsample, blocks in zip(self.downsample_layers, self.res_blocks):
            x = downsample(x)
            x = x.squeeze(2)
            x = blocks(x)
        return x


class ConvPredictor(nn.Module):
    def __init__(self, dims: list[int] | tuple[int, ...]) -> None:
        super().__init__()
        base = int(dims[0])
        hidden = base * 2
        self.net = nn.Sequential(
            nn.Conv2d(base, hidden, kernel_size=2, padding=1),
            ResidualBlock(hidden, spatial_dims=2),
            nn.Conv2d(hidden, base, kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JepaModel(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int = 11,
        dims: list[int] | tuple[int, ...] = (16, 32, 64, 128, 128),
        num_res_blocks: list[int] | tuple[int, ...] = (3, 3, 3, 9, 3),
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(
            in_chans=in_chans,
            dims=list(dims),
            num_res_blocks=list(num_res_blocks),
            num_frames=num_frames,
        )
        self.predictor = ConvPredictor(list(reversed(self.encoder.dims))[:2])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, context: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context_latent = self.encoder(context)
        pred_latent = self.predictor(context_latent)
        with torch.no_grad():
            target_latent = self.encoder(target)
        return pred_latent, target_latent
