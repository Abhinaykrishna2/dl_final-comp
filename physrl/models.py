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
        stem_patch_size: int = 1,
        stem_kernel_size: int = 4,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_frames != 16:
            raise ValueError("this course pipeline expects 16-frame encoder inputs")
        if len(dims) != len(num_res_blocks):
            raise ValueError("dims and num_res_blocks must have the same length")
        if stem_patch_size < 1:
            raise ValueError("stem_patch_size must be >= 1")
        if stem_kernel_size < 1:
            raise ValueError("stem_kernel_size must be >= 1")

        self.embed_dim = int(dims[-1])
        self.dims = [int(dim) for dim in dims]
        self.num_res_blocks = [int(blocks) for blocks in num_res_blocks]
        self.num_frames = int(num_frames)
        self.stem_patch_size = int(stem_patch_size)
        self.stem_kernel_size = int(stem_kernel_size)
        self.drop_path_rate = float(drop_path_rate)

        if self.stem_patch_size == 1:
            stem_conv = nn.Conv3d(
                in_chans,
                self.dims[0],
                kernel_size=(1, self.stem_kernel_size, self.stem_kernel_size),
                padding="same",
            )
        else:
            stem_conv = nn.Conv3d(
                in_chans,
                self.dims[0],
                kernel_size=(1, self.stem_patch_size, self.stem_patch_size),
                stride=(1, self.stem_patch_size, self.stem_patch_size),
            )
        stem = nn.Sequential(stem_conv, LayerNorm(self.dims[0], data_format="channels_first"))
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
        total_blocks = sum(self.num_res_blocks)
        drop_rates = torch.linspace(0, self.drop_path_rate, total_blocks).tolist() if total_blocks > 0 else []
        block_offset = 0
        for idx, channels in enumerate(self.dims):
            spatial_dims = 2 if idx == last_stage else 3
            stage_drop_rates = drop_rates[block_offset : block_offset + self.num_res_blocks[idx]]
            block_offset += self.num_res_blocks[idx]
            self.res_blocks.append(
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels,
                            spatial_dims=spatial_dims,
                            layer_scale_init_value=layer_scale_init_value,
                            drop_path=stage_drop_rates[block_idx],
                        )
                        for block_idx in range(self.num_res_blocks[idx])
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
        stem_patch_size: int = 1,
        stem_kernel_size: int = 4,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(
            in_chans=in_chans,
            dims=list(dims),
            num_res_blocks=list(num_res_blocks),
            num_frames=num_frames,
            stem_patch_size=stem_patch_size,
            stem_kernel_size=stem_kernel_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
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


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        current = int(in_dim)
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
            current = int(hidden_dim)
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SigRegJepaModel(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int = 11,
        dims: list[int] | tuple[int, ...] = (32, 64, 128, 256, 256),
        num_res_blocks: list[int] | tuple[int, ...] = (2, 2, 4, 8, 2),
        num_frames: int = 16,
        stem_patch_size: int = 2,
        stem_kernel_size: int = 4,
        drop_path_rate: float = 0.05,
        layer_scale_init_value: float = 1e-6,
        projector_dim: int = 256,
        projector_hidden_dim: int = 1024,
        projector_layers: int = 3,
        predictor_hidden_dim: int = 1024,
        predictor_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(
            in_chans=in_chans,
            dims=list(dims),
            num_res_blocks=list(num_res_blocks),
            num_frames=num_frames,
            stem_patch_size=stem_patch_size,
            stem_kernel_size=stem_kernel_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.projector = MLPHead(
            self.encoder.embed_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_dim,
            num_layers=projector_layers,
        )
        self.predictor = MLPHead(
            projector_dim,
            hidden_dim=predictor_hidden_dim,
            out_dim=projector_dim,
            num_layers=predictor_layers,
        )

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-1, -2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        embedding = self._pool(features)
        projection = self.projector(embedding)
        return embedding, projection

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        *,
        target_stop_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        context_embedding, context_projection = self._project(context)
        if target_stop_grad:
            with torch.no_grad():
                target_embedding, target_projection = self._project(target)
        else:
            target_embedding, target_projection = self._project(target)

        predicted_projection = self.predictor(context_projection)
        return {
            "context_embedding": context_embedding,
            "target_embedding": target_embedding,
            "context_projection": context_projection,
            "target_projection": target_projection,
            "predicted_projection": predicted_projection,
            "sigreg_embedding": torch.cat([context_embedding, target_embedding], dim=0),
            "sigreg_projection": torch.cat([context_projection, target_projection], dim=0),
        }


class VJepaModel(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int = 11,
        dims: list[int] | tuple[int, ...] = (32, 64, 128, 256, 256),
        num_res_blocks: list[int] | tuple[int, ...] = (2, 2, 4, 8, 2),
        num_frames: int = 16,
        stem_patch_size: int = 1,
        stem_kernel_size: int = 4,
        drop_path_rate: float = 0.05,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        encoder_kwargs = {
            "in_chans": in_chans,
            "dims": list(dims),
            "num_res_blocks": list(num_res_blocks),
            "num_frames": num_frames,
            "stem_patch_size": stem_patch_size,
            "stem_kernel_size": stem_kernel_size,
            "drop_path_rate": drop_path_rate,
            "layer_scale_init_value": layer_scale_init_value,
        }
        self.encoder = ConvEncoder(**encoder_kwargs)
        self.target_encoder = ConvEncoder(**encoder_kwargs)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

        self.predictor = ConvPredictor([self.encoder.embed_dim])
        self.mask_token = nn.Parameter(torch.zeros(1, int(in_chans), 1, 1, 1))

    def train(self, mode: bool = True) -> "VJepaModel":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _apply_input_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 4:
            raise ValueError(f"mask must be shaped (B, 1, H, W), got {tuple(mask.shape)}")
        if mask.shape[0] != x.shape[0] or mask.shape[1] != 1:
            raise ValueError(f"mask batch/channel shape is incompatible with input: {tuple(mask.shape)} vs {tuple(x.shape)}")
        pixel_mask = F.interpolate(mask.float(), size=x.shape[-2:], mode="nearest").to(dtype=torch.bool)
        pixel_mask = pixel_mask.unsqueeze(2)
        return torch.where(pixel_mask, self.mask_token.to(dtype=x.dtype), x)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        momentum = float(momentum)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        masked_context = self._apply_input_mask(context, mask)
        context_features = self.encoder(masked_context)
        predicted_features = self.predictor(context_features)
        with torch.no_grad():
            target_features = self.target_encoder(target)
        return {
            "context_features": context_features,
            "target_features": target_features,
            "predicted_features": predicted_features,
            "mask": mask,
        }
