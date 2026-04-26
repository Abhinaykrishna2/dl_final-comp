from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

def vicreg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sim_coeff: float = 2.0,
    std_coeff: float = 40.0,
    cov_coeff: float = 2.0,
    n_chunks: int = 5,
    num_groups: int = 1,
    fp32_stats: bool = False,
    zscore_for_cov: bool = False,
    adaptive_cov_scale: bool = False,
) -> dict[str, torch.Tensor]:
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"vicreg_loss expects 4D feature maps shaped (B, C, H, W); "
            f"got {tuple(pred.shape)} and {tuple(target.shape)}"
        )

    pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    total = pred.shape[0]

    shuffle_idx = torch.randperm(total, device=pred.device)
    pred = pred[shuffle_idx]
    target = target[shuffle_idx]

    chunks = max(1, int(n_chunks))
    pred_chunks = pred.chunk(chunks, dim=0)
    target_chunks = target.chunk(chunks, dim=0)

    metrics: dict[str, list[torch.Tensor]] = {
        "loss": [],
        "repr_loss": [],
        "std_loss": [],
        "cov_loss": [],
        "std_loss_pred": [],
        "std_loss_target": [],
        "cov_loss_pred": [],
        "cov_loss_target": [],
    }
    for pred_chunk, target_chunk in zip(pred_chunks, target_chunks):
        (
            loss,
            repr_loss,
            std_loss,
            cov_loss,
            std_loss_pred,
            std_loss_target,
            cov_loss_pred,
            cov_loss_target,
        ) = _vicreg_chunk(
            pred_chunk,
            target_chunk,
            sim_coeff=sim_coeff,
            std_coeff=std_coeff,
            cov_coeff=cov_coeff,
            num_groups=num_groups,
            fp32_stats=fp32_stats,
            zscore_for_cov=zscore_for_cov,
            adaptive_cov_scale=adaptive_cov_scale,
        )
        metrics["loss"].append(loss)
        metrics["repr_loss"].append(repr_loss)
        metrics["std_loss"].append(std_loss)
        metrics["cov_loss"].append(cov_loss)
        metrics["std_loss_pred"].append(std_loss_pred)
        metrics["std_loss_target"].append(std_loss_target)
        metrics["cov_loss_pred"].append(cov_loss_pred)
        metrics["cov_loss_target"].append(cov_loss_target)

    return {key: torch.stack(values).mean() for key, values in metrics.items()}


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _vicreg_chunk(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sim_coeff: float,
    std_coeff: float,
    cov_coeff: float,
    num_groups: int,
    fp32_stats: bool,
    zscore_for_cov: bool,
    adaptive_cov_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred.shape[1] % num_groups != 0:
        raise ValueError(
            f"embedding dim {pred.shape[1]} must be divisible by num_groups {num_groups}"
        )

    repr_loss = F.mse_loss(pred, target)
    xs = pred.float() if fp32_stats else pred
    ys = target.float() if fp32_stats else target

    xs = xs - xs.mean(dim=0)
    ys = ys - ys.mean(dim=0)

    std_x = torch.sqrt(xs.var(dim=0, unbiased=False) + 1e-4)
    std_y = torch.sqrt(ys.var(dim=0, unbiased=False) + 1e-4)
    std_loss_pred = torch.mean(F.relu(1.0 - std_x)) / 2.0
    std_loss_target = torch.mean(F.relu(1.0 - std_y)) / 2.0
    std_loss = std_loss_pred + std_loss_target

    if zscore_for_cov:
        xs = xs / std_x.detach().clamp_min(1e-3)
        ys = ys / std_y.detach().clamp_min(1e-3)

    group_width = pred.shape[1] // num_groups
    cov_loss_pred = xs.new_tensor(0.0)
    cov_loss_target = ys.new_tensor(0.0)

    if adaptive_cov_scale:
        scale = min(1.0, float(pred.shape[0]) / float(8 * group_width))
    else:
        scale = 1.0

    denom = max(pred.shape[0] - 1, 1)
    for group_idx in range(num_groups):
        left = group_idx * group_width
        right = left + group_width
        xg = xs[:, left:right]
        yg = ys[:, left:right]
        cov_x = (xg.T @ xg) / denom
        cov_y = (yg.T @ yg) / denom
        cov_loss_pred = cov_loss_pred + _off_diagonal(cov_x).pow_(2).sum().div(group_width)
        cov_loss_target = cov_loss_target + _off_diagonal(cov_y).pow_(2).sum().div(group_width)

    cov_loss_pred = cov_loss_pred / num_groups
    cov_loss_target = cov_loss_target / num_groups
    cov_loss = scale * (cov_loss_pred + cov_loss_target)
    total_loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return (
        total_loss,
        repr_loss.detach(),
        std_loss.detach(),
        cov_loss.detach(),
        std_loss_pred.detach(),
        std_loss_target.detach(),
        cov_loss_pred.detach(),
        cov_loss_target.detach(),
    )


def _same_random_slices(
    *,
    in_dim: int,
    num_slices: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    if device.type == "cuda":
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        slices = torch.randn((in_dim, num_slices), device=device, generator=generator, dtype=dtype)
    else:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        slices = torch.randn((in_dim, num_slices), device=device, generator=generator, dtype=dtype)
    return F.normalize(slices, p=2, dim=0)


def sigreg_loss(
    embeddings: torch.Tensor,
    *,
    num_slices: int = 512,
    num_points: int = 17,
    t_max: float = 3.0,
    seed: int = 0,
    scale_by_samples: bool = True,
    distributed: bool = True,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"sigreg_loss expects embeddings shaped (N, D), got {tuple(embeddings.shape)}")
    if num_slices < 1:
        raise ValueError("num_slices must be >= 1")
    if num_points < 3 or num_points % 2 == 0:
        raise ValueError("num_points must be odd and >= 3")

    x = embeddings.float()
    n_local = torch.tensor(float(x.shape[0]), device=x.device, dtype=x.dtype)
    slices = _same_random_slices(
        in_dim=int(x.shape[1]),
        num_slices=int(num_slices),
        device=x.device,
        dtype=x.dtype,
        seed=seed,
    )
    projection = x @ slices

    t = torch.linspace(0.0, float(t_max), int(num_points), device=x.device, dtype=x.dtype)
    dt = float(t_max) / float(num_points - 1)
    weights = torch.full((num_points,), 2.0 * dt, device=x.device, dtype=x.dtype)
    weights[0] = dt
    weights[-1] = dt
    phi = torch.exp(-0.5 * t.square())
    weights = weights * phi

    args = projection.unsqueeze(-1) * t.view(1, 1, -1)
    cos_sum = torch.cos(args).sum(dim=0)
    sin_sum = torch.sin(args).sum(dim=0)
    n_total = n_local.clone()
    if distributed and dist.is_available() and dist.is_initialized():
        dist.all_reduce(cos_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sin_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_total, op=dist.ReduceOp.SUM)

    denom = n_total.clamp_min(1.0)
    cos_mean = cos_sum / denom
    sin_mean = sin_sum / denom
    err = (cos_mean - phi.view(1, -1)).square() + sin_mean.square()
    statistic = err @ weights
    if scale_by_samples:
        statistic = statistic * denom
    return statistic.mean()


def sigreg_jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigreg_embeddings: torch.Tensor,
    *,
    pred_coeff: float = 1.0,
    sigreg_coeff: float = 0.05,
    num_slices: int = 512,
    num_points: int = 17,
    t_max: float = 3.0,
    seed: int = 0,
    distributed: bool = True,
) -> dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}")

    pred_loss = F.mse_loss(pred.float(), target.float())
    distribution_loss = sigreg_loss(
        sigreg_embeddings,
        num_slices=num_slices,
        num_points=num_points,
        t_max=t_max,
        seed=seed,
        distributed=distributed,
    )
    total = float(pred_coeff) * pred_loss + float(sigreg_coeff) * distribution_loss
    return {
        "loss": total,
        "pred_loss": pred_loss.detach(),
        "repr_loss": pred_loss.detach(),
        "sigreg_loss": distribution_loss.detach(),
    }
