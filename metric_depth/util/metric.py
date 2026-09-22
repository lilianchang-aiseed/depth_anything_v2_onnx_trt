import torch


def _masked_mae(abs_diff, target, low, high):
    mask = (target >= low) & (target < high)
    if not torch.any(mask):
        return float('nan')
    return torch.mean(abs_diff[mask]).item()


def eval_mae_ranges(pred, target, valid_mask, ranges):
    """Return per-range MAE and supporting GT-pixel counts.

    ``ranges`` is an iterable of ``(name, low, high)``. Membership is decided
    from the ground-truth depth, never from the prediction. ``high`` is
    exclusive, so callers can use ``20.000001`` when a 20 m label must be
    included.
    """
    assert pred.shape == target.shape == valid_mask.shape
    finite = valid_mask.bool() & torch.isfinite(pred) & torch.isfinite(target)
    abs_diff = torch.abs(pred - target)
    result = {}
    for name, low, high in ranges:
        mask = finite & (target >= low) & (target < high)
        pixels = int(mask.sum().item())
        result[f'mae_{name}'] = (
            float(abs_diff[mask].mean().item()) if pixels else float('nan')
        )
        result[f'pixels_{name}'] = pixels
    return result


def _masked_silog(diff_log, target, low, high, lambd=0.5):
    mask = (target >= low) & (target < high)
    if not torch.any(mask):
        return float('nan')
    values = diff_log[mask]
    variance = torch.pow(values, 2).mean() - lambd * torch.pow(values.mean(), 2)
    return torch.sqrt(torch.clamp(variance, min=0)).item()


def eval_depth(pred, target, extended=False):
    assert pred.shape == target.shape

    # The caller normally removes invalid pixels. Clamping here keeps ratio/log
    # metrics finite if a model still predicts zero or a negative value.
    eps = torch.finfo(pred.dtype).eps
    pred_safe = torch.clamp(pred, min=eps)
    target_safe = torch.clamp(target, min=eps)

    thresh = torch.max((target_safe / pred_safe), (pred_safe / target_safe))

    d1 = torch.mean((thresh < 1.25).float())
    d2 = torch.mean((thresh < 1.25 ** 2).float())
    d3 = torch.mean((thresh < 1.25 ** 3).float())

    diff = pred - target
    abs_diff = torch.abs(diff)
    relative_error = abs_diff / target_safe
    diff_log = torch.log(pred_safe) - torch.log(target_safe)

    abs_rel = torch.mean(relative_error)
    sq_rel = torch.mean(torch.pow(diff, 2) / target_safe)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred_safe) - torch.log10(target_safe)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    # Keep the default return value identical to the upstream nine-key API.
    metrics = {
        'd1': d1.item(),
        'd2': d2.item(),
        'd3': d3.item(),
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'log10': log10.item(),
        'silog': silog.item(),
    }
    if extended:
        metrics.update({
            'mae': torch.mean(abs_diff).item(),
            'mae_0.2_2m': _masked_mae(abs_diff, target, 0.2, 2.0),
            'mae_2_5m': _masked_mae(abs_diff, target, 2.0, 5.0),
            'mae_5_10m': _masked_mae(abs_diff, target, 5.0, 10.0),
            'mae_10_20m': _masked_mae(abs_diff, target, 10.0, 20.000001),
            'silog_0.2_2m': _masked_silog(diff_log, target, 0.2, 2.0),
            'silog_2_5m': _masked_silog(diff_log, target, 2.0, 5.0),
            'silog_5_10m': _masked_silog(diff_log, target, 5.0, 10.0),
            'silog_10_20m': _masked_silog(
                diff_log, target, 10.0, 20.000001),
            'median_relative_error': torch.median(relative_error).item(),
            'valid_pixel_count': int(target.numel()),
        })
    return metrics
