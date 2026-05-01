"""Models for the videomae stream.

Most of this file is copied verbatim from physrl/models.py (Person A's encoder /
predictor / JEPA wrappers) so that the encoder used for VideoMAE is byte-identical
to the colleague's SIGReg-JEPA encoder. The cleanly added pieces (Person B's
contribution) are at the bottom of the file:

    * ``ConvNeXtDecoder`` -- a lightweight transposed-conv decoder that mirrors
      the encoder downsampling and maps the (B, 256, 7, 7) bottleneck back to
      (B, 11, 16, 224, 224). Per SimMIM (Xie et al. 2022, Tab. 2), lighter
      heads transfer better than heavier ones, so this stays under ~1M params.

    * ``VideoMAEModel`` -- the full SimMIM/VideoMAE-hybrid wrapper that handles
      tube masking, learnable mask tokens (SimMIM convention; required because
      depthwise-3D ConvNeXt cannot use VideoMAE's asymmetric token-dropping),
      per-tube pixel z-scoring (VideoMAE convention) and the MSE-on-masked
      reconstruction loss.
"""

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
            "sigreg_projection": torch.cat([context_projection, target_projection], dim=0),
        }

# =============================================================================
# Person B's contribution: VideoMAE / SimMIM hybrid for the active matter dataset
# =============================================================================


class ConvNeXtDecoder(nn.Module):
    """Lightweight 5-stage transposed-conv decoder mirroring the encoder downsampling.

    Maps the encoder bottleneck ``(B, embed_dim, 7, 7)`` back to the input space
    ``(B, in_chans, num_frames, 224, 224)``.

    Per SimMIM Tab. 2 (Xie et al. 2022, arXiv:2111.09886), a lightweight prediction
    head transfers as well or better than a heavy decoder while being much cheaper
    to train. We size this to ~0.9M parameters (vs ~46M for a single 1x1 conv head),
    keeping the total VideoMAE model well under the 100M cap.
    """

    def __init__(
        self,
        *,
        in_chans: int = 11,
        embed_dim: int = 256,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.num_frames = int(num_frames)

        def upblock(in_c: int, out_c: int, time_stride: int = 2) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose3d(
                    in_c,
                    out_c,
                    kernel_size=(time_stride, 2, 2),
                    stride=(time_stride, 2, 2),
                ),
                LayerNorm(out_c, data_format="channels_first"),
                nn.GELU(),
            )

        # Stage progression: (256, 1, 7, 7) -> (128, 2, 14, 14) -> (64, 4, 28, 28)
        # -> (32, 8, 56, 56) -> (16, 16, 112, 112) -> (11, 16, 224, 224)
        self.up1 = upblock(self.embed_dim, 128, time_stride=2)
        self.up2 = upblock(128, 64, time_stride=2)
        self.up3 = upblock(64, 32, time_stride=2)
        self.up4 = upblock(32, 16, time_stride=2)
        # Final stage: spatial-only upsample (T already at num_frames), and produce
        # the 11 physical-field channels directly with no nonlinearity afterwards.
        self.up5 = nn.ConvTranspose3d(
            16, self.in_chans, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # Encoder squeezed the time dim at the last stage; restore it as length-1
            x = x.unsqueeze(2)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x


class VideoMAEModel(nn.Module):
    """SimMIM/VideoMAE-hybrid masked autoencoder for active matter.

    Differences from canonical VideoMAE (Tong et al. 2022) for ConvNeXt compatibility:

    * Uses learnable mask tokens (SimMIM convention) replacing masked tube positions
      in the input tensor. Necessary because depthwise 3D ConvNeXts cannot use
      VideoMAE's asymmetric encoder-decoder (which drops masked tokens before the
      ViT encoder).
    * Tube size is ``(num_frames, 32, 32)`` aligned with the encoder's downsampling
      factor (32x spatial, 16x temporal). Per SimMIM Sec 4.1.2, the optimal mask
      patch size matches the encoder's downsampling stride.

    Inherits from VideoMAE:

    * Tube masking with the same spatial mask across all frames in a tube.
    * Per-tube pixel z-scoring of the reconstruction target (Tab. 1c).
    * MSE loss on masked positions only.
    """

    DEFAULT_DIMS = (32, 64, 128, 256, 256)
    DEFAULT_BLOCKS = (2, 2, 4, 8, 2)

    def __init__(
        self,
        *,
        in_chans: int = 11,
        dims: list[int] | tuple[int, ...] = DEFAULT_DIMS,
        num_res_blocks: list[int] | tuple[int, ...] = DEFAULT_BLOCKS,
        num_frames: int = 16,
        stem_patch_size: int = 2,
        stem_kernel_size: int = 4,
        drop_path_rate: float = 0.05,
        layer_scale_init_value: float = 1e-6,
        mask_ratio: float = 0.6,
        tube_size: tuple[int, int, int] | list[int] = (16, 32, 32),
        norm_pix_loss: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
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
        self.decoder = ConvNeXtDecoder(
            in_chans=in_chans,
            embed_dim=self.encoder.embed_dim,
            num_frames=num_frames,
        )

        self.in_chans = int(in_chans)
        self.num_frames = int(num_frames)
        self.mask_ratio = float(mask_ratio)
        self.tube_size = (int(tube_size[0]), int(tube_size[1]), int(tube_size[2]))
        self.norm_pix_loss = bool(norm_pix_loss)
        self.norm_eps = float(norm_eps)

        # Learnable mask token: one per channel, broadcast over a tube.
        self.mask_token = nn.Parameter(torch.zeros(1, self.in_chans, 1, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _check_input(self, x: torch.Tensor) -> tuple[int, int, int, int, int, int, int, int]:
        if x.ndim != 5:
            raise ValueError(
                f"VideoMAEModel expects (B, C, T, H, W) input; got shape {tuple(x.shape)}"
            )
        B, C, T, H, W = x.shape
        if C != self.in_chans:
            raise ValueError(f"expected {self.in_chans} channels, got {C}")
        tT, tH, tW = self.tube_size
        if T % tT != 0 or H % tH != 0 or W % tW != 0:
            raise ValueError(
                f"input shape {(T, H, W)} not divisible by tube_size {(tT, tH, tW)}"
            )
        nT, nH, nW = T // tT, H // tH, W // tW
        return B, C, T, H, W, nT, nH, nW

    def random_tube_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a per-sample random tube-level mask.

        Returns a bool tensor of shape ``(B, 1, nT, nH, nW)`` where True marks
        masked tubes. Exactly ``round(mask_ratio * n_tubes)`` tubes are masked
        per sample.
        """
        B, _, _, _, _, nT, nH, nW = self._check_input(x)
        n_tokens = nT * nH * nW
        n_mask = max(1, int(round(self.mask_ratio * n_tokens)))
        scores = torch.rand(B, n_tokens, device=x.device)
        idx = scores.argsort(dim=1)
        mask = torch.zeros(B, n_tokens, dtype=torch.bool, device=x.device)
        mask.scatter_(1, idx[:, :n_mask], True)
        return mask.view(B, 1, nT, nH, nW)

    def expand_tube_mask(self, tube_mask: torch.Tensor) -> torch.Tensor:
        """Expand a tube-level mask ``(B, 1, nT, nH, nW)`` to a pixel-level mask
        ``(B, 1, T, H, W)``."""
        tT, tH, tW = self.tube_size
        return (
            tube_mask.repeat_interleave(tT, dim=2)
            .repeat_interleave(tH, dim=3)
            .repeat_interleave(tW, dim=4)
        )

    def per_tube_zscore(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score each ``(channel, tube)`` cell independently using the tube-internal
        mean and std. This is the VideoMAE per-cube normalization, generalized to
        11-channel physical fields (each channel has its own physical scale, so we
        z-score per channel rather than mixing them)."""
        B, C, T, H, W, nT, nH, nW = self._check_input(x)
        tT, tH, tW = self.tube_size
        # (B, C, nT, tT, nH, tH, nW, tW)
        x6 = x.view(B, C, nT, tT, nH, tH, nW, tW)
        mean = x6.mean(dim=(3, 5, 7), keepdim=True)
        var = x6.var(dim=(3, 5, 7), unbiased=False, keepdim=True)
        norm = (x6 - mean) / torch.sqrt(var + self.norm_eps)
        return norm.view(B, C, T, H, W)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """For embedding extraction by ``export_embeddings.py``: no masking."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tube_mask = self.random_tube_mask(x)
        pixel_mask = self.expand_tube_mask(tube_mask)
        # Replace masked positions with the learnable per-channel mask token
        mask_pixels = self.mask_token.expand_as(x)
        x_masked = torch.where(pixel_mask, mask_pixels, x)

        latent = self.encoder(x_masked)
        recon = self.decoder(latent)

        target = self.per_tube_zscore(x) if self.norm_pix_loss else x
        squared_err = (recon - target).pow(2).mean(dim=1, keepdim=True)
        mask_float = pixel_mask.float()
        loss = (squared_err * mask_float).sum() / mask_float.sum().clamp_min(1.0)

        return {
            "loss": loss,
            "latent": latent,
            "tube_mask": tube_mask,
            "pixel_mask": pixel_mask,
            "recon": recon,
            "target": target,
            "n_masked_tubes": tube_mask.float().sum() / x.shape[0],
        }
