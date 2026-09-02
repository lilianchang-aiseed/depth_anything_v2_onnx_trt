# Distilling DA-Small + RealSense affine into a one-shot metric Depth-Anything-V2

Train a **metric** Depth-Anything-V2 that regresses the `depth_aligned` target from
`make_gt_depthanything.py`. That target is `1 / (s · DA_Small + t)` — Depth-Anything-V2-**Small**
relative disparity scaled by the per-frame RealSense affine. Training a metric net to reproduce it
**distils (DA-Small + affine) into a single network**, so at deploy you run **one monocular forward
pass → metric depth in metres**, with *no RealSense and no SML at inference*.

This is a self-contained add-on to the upstream `Depth-Anything-V2/metric_depth` tree. It reuses the
upstream model, transforms, SiLog loss, and metrics unchanged.

## Files in this add-on

| File | Purpose |
|------|---------|
| `dataset/fisheye_npz.py` | Dataset over your `.npz` exports (returns `image`, `depth`, `valid_mask`). |
| `train_fisheye.py`       | Single-GPU trainer (non-distributed). Split → train → eval → checkpoints → vis. |
| `export_onnx.py`         | Export a trained checkpoint to ONNX for the Jetson TRT path. |

## 0. Prerequisites

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2/metric_depth
pip install -r requirements.txt          # torch, torchvision, opencv-python, ...
# for eval montages / onnx export:
pip install tensorboard onnx

# copy the three add-on files into this directory:
#   dataset/fisheye_npz.py
#   train_fisheye.py
#   export_onnx.py
```

Download the **relative** DA-V2 checkpoint for your chosen encoder (this seeds the DINOv2 encoder; the
metric head trains on top). From the Hugging Face repos `depth-anything/Depth-Anything-V2-{Small,Base,Large}`:

```bash
mkdir -p ../checkpoints
# e.g. Base:
#   ../checkpoints/depth_anything_v2_vitb.pth
# (vits / vitl analogous)
```

> Train on a **desktop/server GPU**, not the Orin NX — `vitb`/`vitl` at 518² won't fit comfortably in
> the NX's memory. The Jetson is the *deploy* target (TRT engine).

## 1. Data preparation

You already produce the training samples with `make_gt_depthanything.py --export-dir <DIR>`. Each
`sample_XXXX.npz` provides everything this trainer needs:

| npz field | used as |
|-----------|---------|
| `left` (H,W,3 uint8 BGR) | input image (converted to RGB internally) |
| `depth_aligned` (H,W float32, NaN outside valid) | **regression target** (metres) |
| `valid_mask` (H,W bool) | supervised pixels (fov & disp>0 & ~sky) |
| `sky_mask` (H,W bool) | only used if you pass `--sky-as-far` |

No conversion step is required — point the trainer at the export directory. It writes a temporally
**chunked** train/val split (`--split-chunk`/`--val-stride`) so near-duplicate consecutive frames
don't leak across the split. To reuse a fixed split, pass `--train-list`/`--val-list` (one `.npz`
path per line) instead of `--data`.

## 2. Training

```bash
python train_fisheye.py \
    --data /home/share/bags/nx-2.0/merged-data_0827-0901-0902 \
    --encoder vits --img-size 518 \
    --min-depth 0.2 --max-depth 20 \
    --bs 4 --epochs 60 --lr 5e-6 --amp \
    --pretrained-from ../checkpoints/depth_anything_v2_vits.pth \
    --save-path exp/fisheye_vits
```

Key flags:

- `--encoder {vits,vitb,vitl}` — see the distillation note below on picking this.
- `--img-size` — network input; **must be a multiple of 14** (518 = 37×14). Your 320² left images are
  upsampled to this. Drop to `322` (23×14) for faster, near-native-resolution training if you're
  compute-bound.
- `--min-depth/--max-depth` — match your GT gating (`dmin=0.2`, `gt_max_depth=20`). The model's output
  is bounded to `(0, max_depth)`.
- `--lr 5e-6` — encoder LR; the head runs at `10×` this, with the upstream poly `(1−it/total)^0.9` decay.
- `--amp` — mixed precision; roughly halves memory/step time on a modern GPU.
- `--pretrained-from ... ` (default, **no** `--load-head`): loads only the DINOv2 encoder from a
  **relative** checkpoint; the metric head trains from init.
- `--pretrained-from <metric ckpt> --load-head`: also warm-starts the metric head (encoder + head)
  from an existing **metric** checkpoint — faster convergence if you have one.
- `--sky-as-far` — weakly labels sky pixels as `max_depth` so the net learns "sky = far" instead of
  leaving sky unsupervised (it will otherwise emit an arbitrary finite depth there at deploy).
- `--resume exp/fisheye_vitb/last.pt` — resume model + optimizer + epoch.

Outputs in `--save-path`:
- `best.pt` / `last.pt` — checkpoints (include `encoder`, `img_size`, `max_depth` for export).
- `train.txt` / `val.txt` — the split actually used.
- `vis/val_*.png` — GT-vs-prediction montages every `--vis-every` epochs.
- TensorBoard logs (unless `--no-tb`): `tensorboard --logdir exp/fisheye_vitb`.

Reported val metrics are the upstream set: `abs_rel, rmse, d1, ...`. `best.pt` tracks lowest `abs_rel`.

### Evaluate a checkpoint only

```bash
python train_fisheye.py --data <DIR> --encoder vitb --img-size 518 \
    --epochs 0 --resume exp/fisheye_vitb/best.pt --save-path exp/eval_only
```

(`--epochs 0` runs the val loop once and writes vis.)

## 3. Deploy (Jetson TRT, monocular, no RealSense/SML)

Export the trained checkpoint to ONNX, then build a TRT engine:

```bash
python export_onnx.py --ckpt exp/fisheye_vitb/best.pt --size 518 \
    --out da2_metric_fisheye_vitb.onnx

trtexec --onnx=da2_metric_fisheye_vitb.onnx \
        --saveEngine=da2_metric_fisheye_vitb.engine --fp16
```

- The ONNX input size is **static** and must be a multiple of 14; the trace is fixed to it (the tracer
  warnings during export are expected for DA-V2). Export at the size you trained, or near your deploy
  size. For your ~7 Hz budget, benchmark `vitb`/`vits` at a smaller square (e.g. `--size 322`) or reuse
  your existing DA-V2 deploy resolution (280×420 is 20×14 by 30×14) — retrain/finetune at that size if
  you deploy far from the training resolution.
- For batched inference across cameras, export with `--dynamic-batch` and add a TRT optimization profile
  (`--minShapes/--optShapes/--maxShapes`), same as your existing DA-V2 engine.

**Inference contract at deploy:** feed an ImageNet-normalized **RGB** tensor at the export size; the
output is metric depth in **metres** at that size. Resize back to 320² for your unprojection. This
replaces the entire `LightStereo → per-frame affine → SML` chain for the monocular path — there is no
`(scale, shift)` fit and no RealSense dependency at run time.

## Notes & caveats

- **Distillation ceiling.** `depth_aligned`'s per-pixel *relative structure* is DA-V2-**Small**'s; the
  RealSense affine only sets two scalars per frame. So the student's relative accuracy asymptotes to
  Small's regardless of student size:
  - `vits` student ≈ reproduces the teacher — pick it if you just want metric-in-one-pass at Small's
    speed/quality.
  - `vitb`/`vitl` student — a larger, often more temporally stable net, but *not* more accurate in
    relative structure than Small. To raise the ceiling, regenerate GT with a Large teacher
    (`--da-model depth-anything/Depth-Anything-V2-Large-hf` in `make_gt_depthanything.py`) **before**
    training, then distil that.
- **The net has no intrinsics input** — it bakes in *this* rectified-left fisheye geometry. It won't
  transfer to a different camera/rig, and a recalibration that changes the rectified geometry means you
  retrain. Fine for a fixed drone rig.
- **Monocular metric depth is less reliable than stereo** for avoidance (it infers scale from learned
  priors and can be confidently wrong on novel textures). Treat this net as a monocular fallback /
  cross-check against the LightStereo→SML path until its held-out `abs_rel`/`d1` earn a swap.
