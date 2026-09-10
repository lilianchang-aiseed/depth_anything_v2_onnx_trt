# Metric Depth Output Head Discussion

這份文件記錄目前針對 Depth Anything V2 metric-depth model output structure 的討論。
內容是設計與實驗規劃；除非另外實作，文件中的 parser 與新 head 並不代表目前程式已經支援。

## 1. User request

目標是修改 DA-V2 metric-depth model，使模型能直接輸出公尺制深度，並研究不同
output parameterization 對下列需求的影響：

- 深度輸出必須為正值。
- 目前部署範圍為 `0.2–20 m`。
- `0.2–15 m` 是 exact metric-depth supervision 的主要範圍。
- `19 m` 是 far/sky pixel 的 supervision threshold 或 far label。
- 近距離物體對 UAV obstacle avoidance 比遠距離更重要。
- 希望比較 Softplus、log-depth 與 inverse-depth/reciprocal structure。
- 保留官方 metric DA-V2 作為 baseline。
- 不應讓原本的 `train_fisheye.py` 或舊 checkpoint 無法執行。
- 不同 model structure、loss 與 hyperparameter 必須能分開記錄及公平比較。

## 2. Current model

官方 relative-depth 與 metric-depth 版本共用幾乎相同的 DINOv2 encoder、四層
intermediate features 與 DPT fusion structure，主要差異在最後輸出語義。

Relative-depth version:

```text
DINOv2 -> DPT fusion -> Conv -> ReLU -> relative output
```

Metric-depth version currently used by this project:

```text
DINOv2 -> DPT fusion -> Conv -> Sigmoid -> multiply max_depth -> depth in metres
```

目前公式為：

```python
depth = max_depth * torch.sigmoid(raw)
```

當 `max_depth=20` 時，prediction 被限制在 `(0, 20) m`。目前 `SiLogLoss` 已在
log-depth domain 比較 prediction 與 GT：

```python
diff_log = torch.log(target) - torch.log(prediction)
```

因此目前已經是「直接輸出 metric depth，使用 log-domain loss」，只是正值與上界
由 `Sigmoid * max_depth` 控制，而不是 Softplus。

Official references:

- [Relative-depth dpt.py](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py)
- [Metric-depth dpt.py](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/depth_anything_v2/dpt.py)
- [Metric-depth README](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md)

## 3. Candidate solutions

### 3.1 Direct bounded depth: current baseline

```python
depth = max_depth * torch.sigmoid(raw)
```

Advantages:

- Simple and officially used by metric DA-V2.
- Prediction cannot exceed `max_depth`.
- Naturally matches the current `far_depth=19`, `max_depth=20` design.
- Existing metric checkpoint semantics remain correct.

Potential problems:

- Depth is parameterized linearly over a large metric range.
- Sigmoid saturates near 0 and `max_depth`.
- It may allocate insufficient output resolution to important near distances.

### 3.2 Direct Softplus depth

```python
depth = min_depth + torch.nn.functional.softplus(raw)
```

Advantages:

- Guarantees positive metric depth.
- Directly returns metres.
- Can continue using SiLog or another log-domain loss.

Potential problems:

- There is no natural upper bound.
- The current one-sided far loss only requires `prediction >= 19`; it does not penalize
  predictions of 50 m, 100 m, or larger.
- An additional upper-range penalty, bounded mapping, or inference clamp would be needed.

This is useful as an experiment, but is not the preferred first replacement for the current
bounded model.

### 3.3 Unbounded Softplus inverse-depth

The initially considered structure was:

```python
inv_depth = torch.nn.functional.softplus(raw) + eps
depth = 1.0 / inv_depth
```

This explicitly includes the reciprocal relation, but it is numerically risky. If
`inv_depth` approaches zero, depth becomes extremely large and the reciprocal derivative is:

```text
d(1/rho) / d(rho) = -1 / rho^2
```

Important operator-precedence detail:

```python
depth = 1.0 / inv_depth + eps      # Wrong protection: eps is added to depth.
depth = 1.0 / (inv_depth + eps)    # eps protects the denominator.
```

The dataset target range alone does not constrain the network prediction range. A model with
unrestricted `raw` values can still produce values far outside the GT range.

### 3.4 Softplus inverse-depth with physical bounds

The accepted inverse-depth bounds are:

```python
rho_min = 1.0 / max_depth   # 1 / 20  = 0.05 1/m
rho_max = 1.0 / min_depth   # 1 / 0.2 = 5.0  1/m
```

A simple Softplus inverse-depth experiment is:

```python
inv_depth = rho_min + torch.nn.functional.softplus(raw)
depth = 1.0 / inv_depth
```

This guarantees `depth <= max_depth` and prevents depth explosion, but prediction may still be
below `min_depth`. One possible bounded version is:

```python
inv_depth = rho_min + torch.nn.functional.softplus(raw)
inv_depth = torch.clamp(inv_depth, max=rho_max)
depth = 1.0 / inv_depth
```

Potential problem: the hard clamp has zero gradient after `inv_depth` exceeds `rho_max`.
Therefore the unclamped training version plus explicit range monitoring may be a better first
Softplus experiment.

### 3.5 Bounded linear inverse-depth

```python
prob = torch.sigmoid(raw)
inv_depth = rho_min + prob * (rho_max - rho_min)
depth = 1.0 / inv_depth
```

This strictly produces `0.2–20 m` and explicitly represents inverse depth. However, with
`raw=0`, it initially predicts approximately `0.396 m`, strongly biasing a randomly initialized
head toward near depth. The last-layer bias should be initialized from a selected typical depth
if this method is tested.

### 3.6 Bounded log inverse-depth: recommended new structure

```python
prob = torch.sigmoid(raw)
log_rho = (
    math.log(rho_min)
    + prob * (math.log(rho_max) - math.log(rho_min))
)
inv_depth = torch.exp(log_rho)
depth = 1.0 / inv_depth
```

Advantages:

- Explicit reciprocal relation.
- Strict `0.2–20 m` output range.
- No arbitrary epsilon and no hard clamp.
- Better numerical behavior across a large distance range.
- At `raw=0`, prediction is approximately the geometric midpoint:
  `sqrt(0.2 * 20) = 2 m`.
- Naturally compatible with the existing SiLog training objective.

Potential problems:

- Sigmoid still saturates near the minimum and maximum range.
- Bounded log inverse-depth is mathematically closely related to bounded log-depth, so any
  improvement does not by itself prove that reciprocal reasoning was learned by the backbone.

## 4. Recommended comparison

The initial experiment should compare:

```text
direct             current max_depth * sigmoid(raw) baseline
softplus-inverse   rho_min + softplus(raw), followed by reciprocal
log-inverse        bounded log inverse-depth, followed by reciprocal
```

Proposed CLI interface:

```bash
--depth-parameterization direct
--depth-parameterization softplus-inverse
--depth-parameterization log-inverse
```

`direct` should remain the default so existing code keeps the official metric behavior.

For a fair architecture comparison, keep all of the following identical:

- Train/validation/test split files.
- Random seed.
- Pretrained DINOv2 backbone.
- Input size and augmentation.
- Batch size and epoch count.
- Optimizer and learning rates.
- Exact metric SiLog loss and far loss.
- `min_depth=0.2`, `metric_depth_max=15`, `far_depth=19`, `max_depth=20`.

Only the depth parameterization should change in the first comparison.

## 5. Initialization and checkpoint compatibility

The final activation layers contain no trainable parameters, so changing an activation may not
cause a state-dict shape error. Nevertheless, the numerical meaning of the final convolution
weights changes completely.

Consequences:

- An old `direct` metric head must not silently be interpreted as `softplus-inverse` or
  `log-inverse`.
- New parameterizations should load the same pretrained encoder/backbone but normally
  reinitialize the output head.
- Avoid `--load-head` when initializing a new parameterization from a relative or direct-metric
  checkpoint, unless an explicit conversion method is implemented.
- Save `depth_parameterization`, `min_depth`, and `max_depth` in `config.yaml`, `best.pt`, and
  `last.pt`.
- Test-only mode must reject a checkpoint whose recorded parameterization differs from the
  selected model.
- Legacy checkpoints without this metadata should be interpreted as `direct` only.

For stricter fairness, initialize the final bias so different parameterizations initially
predict the same typical depth, such as 2 m or the training-set median depth.

## 6. Loss considerations

The existing exact-depth loss can remain SiLog because every candidate returns metric depth:

```text
model -> metric depth in metres -> log() inside SiLogLoss
```

The current one-sided far loss is:

```python
far_loss = relu(far_depth - prediction)
```

It means `19 m or farther`, rather than forcing every far pixel to equal exactly 19 m. This is
well controlled by bounded heads, but requires extra attention for an unbounded Softplus output.

Changing the output parameterization alone may not solve near-object accuracy. A later, separate
experiment can add near-depth weighting or boundary/gradient loss. Do not add it in the first
head comparison, otherwise architecture and loss effects cannot be separated.

Gradient clipping may be useful for reciprocal experiments:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

It should be a recorded hyperparameter and applied consistently when comparing runs.

## 7. Evaluation priorities

Because near obstacles are the deployment priority, compare at least:

```text
val_mae_0.2_2m
val_mae_2_5m
val_abs_rel
val_silog
test_mae_0.2_2m
test_mae_2_5m
far_violation_rate
far_mean_prediction
```

Also inspect the shared-scale visualization:

```text
Input image | GT metric depth | Prediction | shared 0.2–20 m colorbar
```

Per-image min/max normalization should not be used for architecture comparison because it can
hide absolute metric-scale errors.

## 8. Proposed implementation scope

Future implementation should primarily modify:

```text
metric_depth/depth_anything_v2/dpt.py
metric_depth/train_fisheye_with_test.py
```

Recommended internal structure:

1. Let the DPT head return raw logits instead of embedding one fixed final activation.
2. Apply the selected depth parameterization in `DepthAnythingV2.forward()`.
3. Preserve `direct` as the constructor default.
4. Add the CLI parser and pass it into the model constructor.
5. Record and validate the selected mode in checkpoints and experiment config.

`metric_depth/util/loss.py` does not need to change for the first architecture comparison. It
would only need modification when adding weighted near-depth loss or another new loss function.

Before full training, each mode should pass a smoke test that checks:

- finite prediction values;
- expected output shape;
- observed minimum and maximum;
- backward pass without NaN/Inf;
- checkpoint save/reload consistency;
- train, validation, test, and `--model-training false` execution.

## 9. Current recommendation

Use the current `direct` head as the baseline. Test `softplus-inverse` because it represents the
requested Softplus reciprocal hypothesis, but treat `log-inverse` as the leading candidate due to
its bounded output, stable initialization, and compatibility with the current log-domain loss.

Do not conclude that a new head is better from montage appearance alone. Use the same split and
compare near-range metrics, global relative metrics, far-mask behavior, training stability, and
test-set generalization.
