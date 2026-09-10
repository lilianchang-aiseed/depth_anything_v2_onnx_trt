import numpy as np


def _weighted_choice(rng, depths, count, near_sample_weight):
    count = min(int(count), len(depths))
    if count >= len(depths):
        return np.arange(len(depths))
    weights = np.ones(len(depths), dtype=np.float64)
    weights[np.asarray(depths) <= 5.0] = float(near_sample_weight)
    return rng.choice(
        len(depths), count, replace=False, p=weights / weights.sum()
    )


def select_fit_validation_masks(valid, depths, fit_samples,
                                validation_ratio, near_sample_weight, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    ys, xs = np.where(valid)
    count = len(xs)
    if count < 50:
        return None, None

    validation_count = int(round(count * validation_ratio))
    validation_count = min(validation_count, max(0, count - 50))
    validation_mask = np.zeros_like(valid, dtype=bool)
    remaining = np.ones(count, dtype=bool)

    if validation_count:
        selected = _weighted_choice(
            rng, depths[ys, xs], validation_count, near_sample_weight
        )
        validation_mask[ys[selected], xs[selected]] = True
        remaining[selected] = False

    fit_ys, fit_xs = ys[remaining], xs[remaining]
    selected = _weighted_choice(
        rng, depths[fit_ys, fit_xs], fit_samples, near_sample_weight
    )
    fit_mask = np.zeros_like(valid, dtype=bool)
    fit_mask[fit_ys[selected], fit_xs[selected]] = True
    return fit_mask, validation_mask


def robust_affine_invdepth(rel, inv_gt, valid, ransac_iters=200,
                           iters=3, k=2.5, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    ys, xs = np.where(valid)
    if len(xs) < 50:
        return None

    x = rel[ys, xs].astype(np.float64)
    y = inv_gt[ys, xs].astype(np.float64)

    tau = 0.3 * (1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9)
    bs, bt, bi, N = 1.0, 0.0, -1, len(x)

    for _ in range(ransac_iters):
        i, j = rng.integers(0, N, size=2)
        if abs(x[i] - x[j]) < 1e-9:
            continue
        s = (y[i] - y[j]) / (x[i] - x[j])
        t = y[i] - s * x[i]
        inl = int((np.abs(y - (s * x + t)) < tau).sum())
        if inl > bi:
            bs, bt, bi = s, t, inl

    s, t = bs, bt
    keep = np.abs(y - (s * x + t)) < tau

    for _ in range(iters):
        if keep.sum() < 50:
            break
        A = np.stack([x[keep], np.ones(keep.sum())], axis=1)
        sol, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        s, t = float(sol[0]), float(sol[1])
        res = y - (s * x + t)
        med = np.median(res[keep])
        mad = 1.4826 * np.median(np.abs(res[keep] - med)) + 1e-9
        keep = np.abs(res - med) < k * mad

    return s, t, float(keep.mean()), int(keep.sum()), int(N)


def weighted_mse_affine_invdepth(rel, inv_gt, valid, ransac_iters=200,
                                 near_depth=5.0, near_weight=3.0,
                                 rng=None):
    """Select a two-point affine hypothesis by near-weighted inverse-depth MSE."""
    rng = rng if rng is not None else np.random.default_rng(0)
    ys, xs = np.where(valid)
    if len(xs) < 50:
        return None

    x = rel[ys, xs].astype(np.float64)
    y = inv_gt[ys, xs].astype(np.float64)
    depth = 1.0 / np.maximum(y, 1e-12)
    weights = np.ones(len(x), dtype=np.float64)
    weights[depth <= float(near_depth)] = float(near_weight)
    weight_sum = weights.sum()

    best_s, best_t = None, None
    best_mse = np.inf
    candidate_count = 0
    for _ in range(int(ransac_iters)):
        i, j = rng.integers(0, len(x), size=2)
        if i == j or abs(x[i] - x[j]) < 1e-9:
            continue
        s = (y[i] - y[j]) / (x[i] - x[j])
        if not np.isfinite(s) or s <= 0:
            continue
        t = y[i] - s * x[i]
        prediction = s * x + t
        mse = float(np.sum(weights * (prediction - y) ** 2) / weight_sum)
        candidate_count += 1
        if mse < best_mse:
            best_s, best_t, best_mse = float(s), float(t), mse

    if best_s is None:
        return None
    return best_s, best_t, best_mse, candidate_count, int(len(x))


def _pava_increasing(values, weights):
    levels, masses, starts, ends = [], [], [], []
    for index, (value, weight) in enumerate(zip(values, weights)):
        levels.append(float(value))
        masses.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            mass = masses[-2] + masses[-1]
            level = (
                levels[-2] * masses[-2] + levels[-1] * masses[-1]
            ) / mass
            levels[-2:] = [level]
            masses[-2:] = [mass]
            ends[-2:] = [ends[-1]]
            starts.pop()

    output = np.empty(len(values), dtype=np.float64)
    for level, start, end in zip(levels, starts, ends):
        output[start:end] = level
    return output


def _pchip_slopes(x, y):
    if len(x) == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return np.array([slope, slope], dtype=np.float64)

    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    same = delta[:-1] * delta[1:] > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    interior = np.flatnonzero(same) + 1
    slopes[interior] = (
        (w1[same] + w2[same]) /
        (w1[same] / delta[:-1][same] + w2[same] / delta[1:][same])
    )

    def endpoint(h0, h1, d0, d1):
        value = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(value) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(value) > abs(3.0 * d0):
            return 3.0 * d0
        return value

    slopes[0] = endpoint(h[0], h[1], delta[0], delta[1])
    slopes[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def _pchip_evaluate(x, y, query):
    query = np.asarray(query, dtype=np.float64)
    clipped = np.clip(query, x[0], x[-1])
    index = np.clip(
        np.searchsorted(x, clipped, side="right") - 1, 0, len(x) - 2
    )
    h = x[index + 1] - x[index]
    u = (clipped - x[index]) / h
    slopes = _pchip_slopes(x, y)
    return (
        (2*u**3 - 3*u**2 + 1) * y[index]
        + (u**3 - 2*u**2 + u) * h * slopes[index]
        + (-2*u**3 + 3*u**2) * y[index + 1]
        + (u**3 - u**2) * h * slopes[index + 1]
    )


def isotonic_pchip_invdepth(rel, inv_gt, fit_mask, metric_depth_max):
    affine = robust_affine_invdepth(rel, inv_gt, fit_mask)
    if affine is None:
        return None

    s, t, _, _, sample_count = affine
    ys, xs = np.where(fit_mask)
    x = rel[ys, xs].astype(np.float64)
    y = inv_gt[ys, xs].astype(np.float64)

    residual = y - (s * x + t)
    median = np.median(residual)
    mad = 1.4826 * np.median(np.abs(residual - median)) + 1e-9
    keep = np.abs(residual - median) < 2.5 * mad
    x, y = x[keep], y[keep]
    if len(x) < 50:
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]
    bins = np.array_split(np.arange(len(x)), min(128, len(x)))
    knot_x = np.array([np.median(x[idx]) for idx in bins])
    knot_y = np.array([np.median(y[idx]) for idx in bins])
    knot_w = np.array([len(idx) for idx in bins], dtype=np.float64)

    unique_x, inverse = np.unique(knot_x, return_inverse=True)
    if len(unique_x) < 2:
        return None

    merged_y = np.zeros(len(unique_x), dtype=np.float64)
    merged_w = np.zeros(len(unique_x), dtype=np.float64)
    np.add.at(merged_y, inverse, knot_y * knot_w)
    np.add.at(merged_w, inverse, knot_w)
    merged_y /= merged_w

    monotone_y = _pava_increasing(merged_y, merged_w)
    mapped_inv = _pchip_evaluate(unique_x, monotone_y, rel)

    threshold_inv = 1.0 / float(metric_depth_max)
    unique_y, first = np.unique(monotone_y, return_index=True)

    if threshold_inv <= unique_y[0]:
        da_far_threshold = unique_x[0]
    elif threshold_inv >= unique_y[-1]:
        da_far_threshold = unique_x[-1]
    else:
        da_far_threshold = float(
            np.interp(threshold_inv, unique_y, unique_x[first])
        )

    return mapped_inv, {
        "mode": "isotonic-pchip",
        "s": s,
        "t": t,
        "inl": float(keep.mean()),
        "inlier_count": int(keep.sum()),
        "fit_sample_count": int(sample_count),
        "pchip_knots": int(len(unique_x)),
        "da_min_anchor": float(unique_x[0]),
        "da_max_anchor": float(unique_x[-1]),
        "da_far_threshold": float(da_far_threshold),
    }


def validation_depth_statistics(pred_depth, target_depth, validation_mask):
    valid = (
        validation_mask
        & np.isfinite(pred_depth)
        & np.isfinite(target_depth)
        & (target_depth > 0)
    )

    result = {
        "validation_count": int(validation_mask.sum()),
        "validation_valid_count": int(valid.sum()),
        "val_mae_m": np.nan,
        "val_rmse_m": np.nan,
        "val_median_relative_error": np.nan,
        "val_mae_lt5m": np.nan,
        "val_mae_0_2m": np.nan,
        "val_mae_2_5m": np.nan,
        "val_mae_5_10m": np.nan,
        "val_mae_10_20m": np.nan,
    }

    if not valid.any():
        return result

    target = target_depth[valid].astype(np.float64)
    error = np.abs(pred_depth[valid].astype(np.float64) - target)

    result.update({
        "val_mae_m": float(error.mean()),
        "val_rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "val_median_relative_error": float(np.median(error / target)),
    })

    selected = target < 5.0
    if selected.any():
        result["val_mae_lt5m"] = float(error[selected].mean())

    for key, lower, upper in (
        ("val_mae_0_2m", 0.0, 2.0),
        ("val_mae_2_5m", 2.0, 5.0),
        ("val_mae_5_10m", 5.0, 10.0),
        ("val_mae_10_20m", 10.0, 20.0),
    ):
        selected = (target >= lower) & (target < upper)
        if selected.any():
            result[key] = float(error[selected].mean())

    return result


def step3_fit_metric_L(da_L, rs_depth_L, anchor_valid,
                       da_metric=False,
                       fit_mode="isotonic-pchip",
                       metric_depth_max=15.0,
                       fit_samples=10000,
                       near_sample_weight=3.0,
                       validation_ratio=0.2):
    valid_anchor = (
        anchor_valid
        & np.isfinite(da_L)
        & (da_L > 0)
        & np.isfinite(rs_depth_L)
        & (rs_depth_L > 0)
    )

    rng = np.random.default_rng(0)
    fit_mask, validation_mask = select_fit_validation_masks(
        valid_anchor, rs_depth_L, fit_samples,
        validation_ratio, near_sample_weight, rng=rng
    )

    if fit_mask is None or fit_mask.sum() < 50:
        return None, {}

    common_info = {
        "anchor_count": int(valid_anchor.sum()),
        "fit_sample_count": int(fit_mask.sum()),
    }

    if da_metric:
        ratio = float(np.median(rs_depth_L[fit_mask] / da_L[fit_mask]))
        out = (da_L * ratio).astype(np.float32)
        out[~np.isfinite(out) | (out <= 0)] = np.nan
        info = {
            "mode": "metric",
            "ratio": ratio,
            "inl": 1.0,
            "inlier_count": int(fit_mask.sum()),
            **common_info,
        }
        info.update(validation_depth_statistics(
            out, rs_depth_L, validation_mask
        ))
        return out, info

    inv_gt = np.zeros_like(rs_depth_L)
    inv_gt[valid_anchor] = 1.0 / rs_depth_L[valid_anchor]

    if fit_mode == "weighted-mse":
        fit = weighted_mse_affine_invdepth(
            da_L, inv_gt, fit_mask,
            near_weight=near_sample_weight, rng=rng
        )
        if fit is None:
            return None, {}
        s, t, weighted_mse, candidate_count, fit_sample_count = fit
        ga_inv = s * da_L + t
        da_far_threshold = (
            (1.0 / metric_depth_max - t) / s
            if s > 0 else float(np.nanmin(da_L[fit_mask]))
        )
        curve_info = {
            "mode": "weighted-mse",
            "s": s,
            "t": t,
            "inl": np.nan,
            "inlier_count": 0,
            "fit_sample_count": fit_sample_count,
            "weighted_invdepth_mse": weighted_mse,
            "candidate_count": candidate_count,
            "near_depth": 5.0,
            "near_weight": float(near_sample_weight),
            "da_far_threshold": float(da_far_threshold),
        }
    elif fit_mode == "isotonic-pchip":
        curved = isotonic_pchip_invdepth(
            da_L, inv_gt, fit_mask, metric_depth_max
        )
        if curved is None:
            return None, {}
        ga_inv, curve_info = curved
    else:
        fit = robust_affine_invdepth(da_L, inv_gt, fit_mask, rng=rng)
        if fit is None:
            return None, {}
        s, t, inl, inlier_count, fit_sample_count = fit
        ga_inv = s * da_L + t
        da_far_threshold = (
            (1.0 / metric_depth_max - t) / s
            if s > 0
            else float(np.nanmin(da_L[fit_mask]))
        )
        curve_info = {
            "mode": "affine",
            "s": s,
            "t": t,
            "inl": inl,
            "inlier_count": inlier_count,
            "fit_sample_count": fit_sample_count,
            "da_far_threshold": float(da_far_threshold),
        }

    good = np.isfinite(ga_inv) & (ga_inv > 0)
    depth_L = np.full_like(da_L, np.nan, dtype=np.float32)
    depth_L[good] = (1.0 / ga_inv[good]).astype(np.float32)

    info = {
        **curve_info,
        **common_info,
        "coverage": float(good.mean()),
    }
    info.update(validation_depth_statistics(
        depth_L, rs_depth_L, validation_mask
    ))
    return depth_L, info
