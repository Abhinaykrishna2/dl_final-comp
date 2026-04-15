from __future__ import annotations

import torch
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
