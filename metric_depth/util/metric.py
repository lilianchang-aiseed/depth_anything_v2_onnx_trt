import torch


def _masked_mae(abs_diff, target, low, high):
    mask = (target >= low) & (target < high)
    if not torch.any(mask):
        return float('nan')
    return torch.mean(abs_diff[mask]).item()


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
            'median_relative_error': torch.median(relative_error).item(),
            'valid_pixel_count': int(target.numel()),
        })
    return metrics
