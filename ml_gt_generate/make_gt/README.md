# SML Ground-Truth Generation

`make_gt_depthanything.py` reads a ROS 2 bag directly, projects RealSense
metric depth into a rectified stereo-left image, and uses those projected
points to align Depth Anything V2 (DA-V2) relative inverse depth. It exports a
dense metric pseudo-ground-truth dataset, diagnostic montage images, and one
CSV row of fitting statistics per exported frame.

For a direct single-bag run, output image and NPZ names are derived from the
bag path. For a bag under `.../0828/train_1_0_20260828_161432/`, the first pair
is:

```text
0828_train_1_0_20260828_161432_1.png
0828_train_1_0_20260828_161432_1.npz
```

`run_all_bags.sh` adds a stable manifest number, such as `bag01`, to make
flattened multi-bag output names unique.

The bag is read offline with `rosbags.AnyReader`; do **not** run
`ros2 bag play`.

## Current commands

Run these commands from the SML project root, for example `~/Projects/sml`.

### Full 2026-08-27 dataset

```bash
conda activate aiseed1
cd ~/Projects/sml

python3 make_gt/make_gt_depthanything.py \
  --bag /home/share/bags/nx-2.0/0827/train_1_0_20260827_164533/train_1_0_20260827_164533_0.mcap \
  --gt-config make_gt/gt_config_0827.yaml \
  --vis-dir out_data/0827_164533_gt/vis \
  --export-dir out_data/0827_164533_gt/dataset \
  --max-pairs 1691
```

The current 0827 config already supplies these fitting defaults, so they do
not need to be repeated on the command line:

```text
--fit-samples 10000
--near-sample-weight 3.0
--ransac-validation-ratio 0.2
```

### Ten-frame smoke test

This runs inference on every 100th disparity frame and exports at most ten
successful samples:

```bash
python3 make_gt/make_gt_depthanything.py \
  --bag /home/share/bags/nx-2.0/0827/train_1_0_20260827_164533/train_1_0_20260827_164533_0.mcap \
  --gt-config make_gt/gt_config_0827.yaml \
  --vis-dir out_data/0827_164533_smoke/vis \
  --export-dir out_data/0827_164533_smoke/dataset \
  --max-pairs 10 \
  --frame-interval 100
```

When all selected frames are valid, their source disparity indices are
`0, 100, 200, ..., 900`. If one is rejected because of synchronization or
fitting failure, the script continues to the next interval until it exports
ten samples or reaches the end of the bag.

`--frame-interval` reduces DA-V2 inference work, but the current implementation
still loads all requested bag topics into memory before processing.

### Isotonic-PCHIP clamp dataset

`make_gt_depthanything_clamp.py` leaves the original generator unchanged. Its
defaults are `--fit-mode isotonic-pchip --metric-depth-max 15 --far-depth 19
--model-max-depth 20`. Exact depths through 15 m use `metric_valid_mask`;
farther pixels, low relative-DA values beyond the fitted 15 m threshold, and
sky use `far_mask`. The stored 19 m value means "farther than 15 m", not an
exact 19 m measurement.

Use the same command as above and replace the script filename. The clamp NPZ
additionally stores `metric_valid_mask`, `far_mask`, `sky_mask`, `da_relative`,
and the three depth thresholds.

## Required inputs

### 1. Python environment

The environment must provide:

- `rosbags`
- `numpy`
- `opencv-python`
- `torch`
- `transformers`
- `Pillow`
- `PyYAML`
- `natsort` (used by `postprocess_review.py`)
- the Python `ncnn` binding when NCNN sky segmentation is enabled

CUDA is used automatically when PyTorch reports that it is available.

### 2. ROS 2 bag

Required topics for the current 0827 configuration:

| Role | Topic | Expected message |
|---|---|---|
| Rectified RGB input | `/stereo_1_0/left/image_rect` | `sensor_msgs/msg/Image`, normally `bgr8` or `rgb8` |
| Stereo frame/timestamp anchor | `/stereo_1_0/disparity` | `sensor_msgs/msg/Image`, raw `32FC1` disparity |
| Primary metric depth | `/d455/d455_node/depth/image_rect_raw` | `sensor_msgs/msg/Image`, normally `16UC1` millimetres |
| D455 intrinsics | `/d455/d455_node/depth/camera_info` | `sensor_msgs/msg/CameraInfo` |
| Optional secondary depth | `/d435/d435_node/depth/image_rect_raw` | `sensor_msgs/msg/Image`, normally `16UC1` millimetres |
| Optional D435 intrinsics | `/d435/d435_node/depth/camera_info` | `sensor_msgs/msg/CameraInfo` |

The left image, D455 depth, and disparity topics are mandatory. D435 is
disabled automatically when its depth topic has no messages. CameraInfo may
fall back to calibration intrinsics when it is unavailable.

Supported image encodings are `16UC1`, `mono16`, `mono8`, `8UC1`, `32FC1`,
`rgb8`, `bgr8`, `rgba8`, and `bgra8`.

### 3. GT configuration

Always select the configuration matching the physical rig and target camera:

```text
make_gt/gt_config_0827.yaml  -> /stereo_1_0/left/image_rect
make_gt/gt_config_0828.yaml  -> /stereo_2_1/left/image_rect
```

Command-line values override the GT YAML, which overrides legacy Python
defaults. Paths inside a GT YAML are resolved relative to that YAML file.

The config supplies:

- topic names;
- raw or Kalibr-compatible camera-chain paths;
- depth scales;
- FOV mask;
- synchronization and depth ranges;
- DA-V2 model;
- sky segmentation settings;
- RANSAC sampling settings;
- legacy quality-control thresholds.

### 4. Calibration

The script internally needs these directions:

```text
T_Left<-D455
T_D455<-D435
```

`correct_matrix_direction.py` reads a two-camera Kalibr `*-camchain.yaml` or
`*-results-cam.txt` and normalizes the matrix direction automatically.

The current 0827 config uses:

```text
calib/2026-08-27/stereo_d455_d435_extrinsic/
├── d455-left/stereo_1_0_d455_composed-camchain.yaml
└── d455-d435/flight_data_2026_08_27-10_27_07_0-camchain.yaml
```

The first 0827 file is Kalibr-compatible but was composed through a shared IMU;
it is not a direct two-camera Kalibr estimate. The second is the direct
D455-infra1/D435-infra1 Kalibr chain.

### 5. FOV and sky models

The optional deploy mask is a 2-D `.npy` boolean or 0/1 array. It is resized
with nearest-neighbour interpolation when its shape differs from the target
image. Without a deploy mask, the full target image is used.

NCNN sky segmentation needs both a `.param` and `.bin` file. The current
configs use `EGE_165.ncnn.param`, `EGE_165.ncnn.bin`, input blob `in0`, output
blob `out0`, input size 384, and guided refinement.

## Processing workflow

### 1. Offline loading and synchronization

1. Read only the configured topics from the MCAP/ROS 2 bag.
2. Sort every topic by `header.stamp`.
3. Use each selected disparity timestamp as the frame anchor.
4. Find the nearest left image, D455 depth, and optional D435 depth within
   `--sync-tol`, after applying configured time offsets.
5. Reject a frame when a mandatory synchronized message is missing.

### 2. RGB to DA-V2 relative inverse depth

```text
rectified stereo-left image
        -> RGB conversion
        -> DA-V2 image processor
        -> DA-V2 inference
        -> bicubic resize to the original left-image shape
        -> dense relative inverse-depth/disparity-like output
```

For the current stereo topics, the final DA-V2 grid is 320 x 320. It is not yet
in metres.

### 3. D455 depth to the stereo-left grid

```text
D455 raw depth
        -> multiply by depth_scale (normally 0.001 mm-to-m)
        -> back-project with D455 intrinsics
        -> transform 3-D points with T_Left<-D455
        -> project with target intrinsics/distortion
        -> z-buffer: keep the nearest source point per target pixel
        -> optional 2 x 2 splat to close sub-pixel projection holes
```

The result is a sparse metric-depth map aligned to the left image.

### 4. Optional D435 projection and merge

D435 follows the same back-project/transform/project process. The two
left-grid depth maps are then merged:

- `fill`: D455 wins where it is valid; D435 only fills D455 holes.
- `min`: where both are valid, keep the nearer depth; also retain pixels valid
  in only one sensor.

The montage reports only the merged coverage. Individual D455 and D435
coverage values are retained in the CSV.

### 5. Anchor filtering and sky removal

An anchor is eligible when it:

- contains finite projected RealSense depth;
- is deeper than `dmin` and shallower than `max_fit_depth`;
- lies inside the deploy FOV mask;
- is not classified as sky.

LightStereo disparity validity does not currently control GT validity.

### 6. Near-weighted fitting and validation split

All valid anchors are divided into disjoint sets:

1. Select `ransac_validation_ratio` (default 20%) as held-out validation.
2. Select at most `fit_samples` (default 10000) from the remaining anchors.
3. In both selections, anchors at 5 m or nearer receive
   `near_sample_weight` (default 3.0); farther anchors receive weight 1.0.

Validation anchors never participate in fitting.

### 7. Per-frame RANSAC global alignment

For relative DA-V2 output, fitting uses inverse metric depth:

```text
1 / RealSense_depth(u,v)
    ~= scale * DA_relative(u,v) + shift
```

RANSAC repeatedly proposes a scale/shift from two sampled anchors, selects the
model supported by the most inliers, and performs robust least-squares
refinement. The one scale and one shift are shared by the entire frame:

```text
aligned_inverse_depth = scale * DA_relative + shift
metric_depth = 1 / aligned_inverse_depth
```

The output is marked invalid rather than clipped when aligned inverse depth is
non-positive or the resulting depth exceeds `gt_max_depth`.

When `--da-metric` is active, the script instead estimates one robust median
depth ratio; scale/shift RANSAC is not used.

### 8. Validation, QC, and export

The held-out anchors measure metric MAE, RMSE, median relative error, and MAE in
four distance bands. These new errors are recorded but do not yet move files
into `_review`.

The current legacy QC sends an output to `_review` when any of these occurs:

- too few valid dense output pixels;
- `depth_p90 / depth_p10` is below `qc_min_spread`;
- absolute scale is below `qc_min_s`;
- fitting inlier ratio is below `qc_min_inl`.

`QC` means **Quality Control**. A review result is suspicious, not necessarily
wrong.

## Outputs

### Batch output

`run_all_bags.sh` recursively finds every MCAP and writes all samples into two
shared flat directories. Each discovered bag receives a persistent number in
`bag_manifest.csv`:

```bash
bash make_gt/run_all_bags.sh \
  --bag-dir "$COMMON_SHARE/bags/nx-2.0/0901" \
  --gt-config make_gt/gt_config_0828.yaml \
  --out-dir out_data/0901_batch
```

```text
out_data/0901_batch/
├── dataset/
│   ├── 0901_session_A_bag01_1.npz
│   └── _review/
├── vis/
│   ├── 0901_session_A_bag01_1.png
│   └── _review/
├── statistics/
│   └── 0901_session_A_bag01.csv
├── logs/
│   └── 0901_session_A_bag01.log
├── completed/
│   └── 0901_session_A_bag01.complete
└── bag_manifest.csv
```

Existing CSV, PNG, and NPZ files are never overwritten. A `.complete` marker
is created only after one bag succeeds; completed bags are skipped on later
runs. Existing partial output without a completion marker is reported as
`INCOMPLETE` and left untouched.

### Postprocess into `_review`

`postprocess_review.py` reads the per-bag CSV files and moves matching NPZ/PNG
pairs into their respective `_review` directories. It is a dry run unless
`--apply` is supplied. `--mae-rule both` requires both MAE limits to fail;
`PREFIX=140` reviews frames 1 through 139 and keeps frame 140.

```bash
python3 make_gt/postprocess_review.py \
  --out-dir out_data/BATCH \
  --mae-0-2-max 1 --mae-2-5-max 2 --mae-rule both \
  --dirty-before 0827_session_bag01=140
```

After checking the preview, repeat the command with `--apply`. An audit CSV
named `postprocess_review_YYYYMMDD_HHMMSS.csv` is created in the batch root.

### Direct single-bag output

Given:

```text
--vis-dir out_data/0827_164533_gt/vis
--export-dir out_data/0827_164533_gt/dataset
```

the output layout is:

```text
out_data/0827_164533_gt/
├── vis/
│   ├── 0827_train_1_0_20260827_164533.csv
│   ├── 0827_train_1_0_20260827_164533_1.png
│   ├── 0827_train_1_0_20260827_164533_2.png
│   └── _review/
│       └── 0827_train_1_0_20260827_164533_N.png
└── dataset/
    ├── 0827_train_1_0_20260827_164533_1.npz
    ├── 0827_train_1_0_20260827_164533_2.npz
    └── _review/
        └── 0827_train_1_0_20260827_164533_N.npz
```

Numbers start at 1 and are sequential successful-output indices. The original
disparity index is stored as `source_frame` in the CSV.

### NPZ dataset fields

| Key | Type | Meaning |
|---|---|---|
| `depth_aligned` | `float32` array | Dense globally aligned metric depth in metres; invalid pixels are `NaN`. |
| `realsense_depth_merged` | `float32` array | D455/D435 metric depth merged after projection into the stereo-left grid; missing pixels are `NaN`. |
| `left` | image array | Decoded stereo-left image used by DA-V2; color arrays use OpenCV BGR order. |
| `stamp` | scalar `float64` | Disparity anchor timestamp in seconds. |

### Visualization montage

The montage contains:

1. rectified left image;
2. DA-V2 relative output;
3. raw D455 metric depth;
4. merged RealSense depth projected into the left grid;
5. globally aligned dense metric depth with scale, shift, sample count, and
   fitting inlier ratio;
6. sky segmentation overlay;
7. final masked GT with held-out near-depth MAE, median relative error, and QC
   status;
8. LightStereo disparity;
9. an additional vertical GT/disparity comparison.

Depth visualization uses the configured `dmin` to `dmax` color range. The
visualization does not replace the raw floating-point values stored in NPZ.

### CSV fields

Each bag has its own statistics CSV. Creation is exclusive: the program stops
instead of replacing an existing CSV. Ratios are stored as fractions, so
`0.25` means 25%.

| Column | Meaning |
|---|---|
| `frame` | One-based sequential successful-output number used in the PNG/NPZ filename. |
| `source_frame` | Original disparity-message index before interval sampling or frame rejection. |
| `stamp` | Disparity anchor timestamp in seconds. |
| `scale` | Per-frame global inverse-depth scale; `NaN` in DA-metric ratio mode. |
| `shift` | Per-frame global inverse-depth shift; `NaN` in DA-metric ratio mode. |
| `anchor_count` | All eligible aligned RealSense anchors before fitting/validation sampling. |
| `sample_count` | Anchors actually used by RANSAC, capped by `fit_samples`. |
| `inlier_count` | Fitting samples retained by the final robust model. |
| `inlier_ratio` | `inlier_count / sample_count`; this measures fitting consistency, not calibrated probability. |
| `validation_count` | Held-out anchors that never participated in fitting. |
| `validation_valid_count` | Held-out anchors for which the aligned prediction remains finite and within the output depth range. |
| `D455_coverage` | Fraction of the target image receiving projected D455 depth before FOV/sky filtering. |
| `D435_coverage` | Fraction receiving projected D435 depth before FOV/sky filtering; zero when D435 is disabled or missing. |
| `merged_coverage` | Fraction receiving valid depth after D455/D435 merge. |
| `GT_valid_ratio` | Fraction of the target image retained in final `valid_mask`. |
| `val_MAE_all_m` | Mean absolute metric-depth error on valid held-out anchors, in metres. |
| `val_RMSE_all_m` | Root mean squared metric-depth error on held-out anchors, in metres; more sensitive to large errors. |
| `val_median_relative_error` | Median of `abs(pred-target)/target` on held-out anchors. |
| `val_MAE_0_2m` | Held-out MAE for target depths from 0 to below 2 m. |
| `val_MAE_2_5m` | Held-out MAE for target depths from 2 to below 5 m. |
| `val_MAE_5_10m` | Held-out MAE for target depths from 5 to below 10 m. |
| `val_MAE_10_20m` | Held-out MAE for target depths from 10 to below 20 m. |
| `depth_p10` | 10th percentile of valid generated metric depth. |
| `depth_p50` | Median valid generated metric depth. |
| `depth_p90` | 90th percentile of valid generated metric depth. |
| `QC_status` | `PASS` or `REVIEW` from the current legacy QC rules. |
| `QC_reason` | Semicolon-separated rule(s) that caused `REVIEW`; empty for `PASS`. |

Validation statistics evaluate only positions where projected RealSense depth
exists. They cannot directly measure local DA-V2 error outside RealSense
coverage.

## Command-line parsers

Defaults come from the selected GT config when present. The values below
describe the parser role and the current 0827 behavior.

### Core I/O and sampling

| Parser | Purpose |
|---|---|
| `-h`, `--help` | Print parser help. |
| `--gt-config PATH` | Select the dated runtime/configuration YAML. This is not a raw Kalibr file. |
| `--bag PATH` | MCAP file or ROS 2 bag path read offline. |
| `--vis-dir PATH` | Montage, `_review`, and statistics CSV directory. |
| `--export-dir PATH` | NPZ dataset directory; omit to disable NPZ export. |
| `--output-prefix NAME` | Safe PNG/NPZ filename prefix. Batch mode supplies `DATE_SESSION_bagNN`; direct mode defaults to `DATE_SESSION`. |
| `--stats-csv PATH` | Per-bag statistics CSV path. Default: `VIS_DIR/OUTPUT_PREFIX.csv`. |
| `--max-pairs N` | Maximum number of successful outputs. Current config default: 1000. |
| `--frame-interval N` | Process every Nth disparity frame. Default: 1. |
| `--max-pairs-interval N` | Alias for `--frame-interval`. |
| `--tile N` | Square montage tile size in pixels. Current default: 300. |

### Topic overrides

| Parser | Purpose |
|---|---|
| `--left-topic TOPIC` | Rectified target/DA-V2 image topic. |
| `--disp-topic TOPIC` | Raw LightStereo disparity topic and frame timestamp anchor. |
| `--d455-depth TOPIC` | Primary D455 raw metric-depth topic. |
| `--d455-info TOPIC` | D455 depth CameraInfo topic. |
| `--d435-depth TOPIC` | Optional D435 depth topic. |
| `--d435-info TOPIC` | Optional D435 CameraInfo topic. |

### Calibration and projection

| Parser | Purpose |
|---|---|
| `--left-calib PATH` | Raw/Kalibr-compatible left-D455 camchain YAML or results TXT. |
| `--d435-calib PATH` | Raw D455-D435 Kalibr camchain YAML or results TXT. |
| `--left-proj fx fy cx cy` | Override target-left projection intrinsics. |
| `--left-dist k1 k2 p1 p2` | Override target-left radtan distortion. |
| `--d455-proj fx fy cx cy` | Override D455 depth/infra1 intrinsics. |
| `--d435-proj fx fy cx cy` | Override D435 source intrinsics. |
| `--d435-source color\|infra1\|kalibr_infra1` | Select D435 transform/source path. Current dated configs use `kalibr_infra1`. |
| `--d455-d2c-R r00 ... r22` | Override legacy D455 Depth-to-Color rotation, row-major. Used only by applicable color/infra1 transform paths. |
| `--d455-d2c-t tx ty tz` | Override legacy D455 Depth-to-Color translation. |
| `--d435-d2c-R r00 ... r22` | Override legacy D435 Depth-to-Color rotation. |
| `--d435-d2c-t tx ty tz` | Override legacy D435 Depth-to-Color translation. |

### FOV, DA-V2, synchronization, and fitting

| Parser | Purpose |
|---|---|
| `--deploy-mask PATH` | Optional `.npy` FOV mask. |
| `--no-deploy-mask` | Ignore the config mask and use the whole target image. |
| `--da-model ID_OR_PATH` | Hugging Face model identifier or local DA-V2 model directory. |
| `--da-metric` | Treat DA output as metric depth and fit one median multiplicative ratio. |
| `--no-da-metric` | Treat DA output as relative inverse depth and fit scale plus shift. Current default. |
| `--depth-scale VALUE` | D455 raw-depth conversion to metres. Current default: 0.001. |
| `--d435-depth-scale VALUE` | D435 raw-depth conversion to metres. Current default: 0.001. |
| `--sync-tol SEC` | Maximum absolute nearest-message time difference. Current default: 0.05 s. |
| `--depth-time-offset SEC` | D455 timestamp offset applied during matching. Current default: 0.0. |
| `--d435-time-offset SEC` | D435 timestamp offset. Current default: 0.0. |
| `--dmin METRES` | Minimum accepted anchor depth and visualization color minimum. Current default: 0.2. |
| `--dmax METRES` | Visualization color maximum; it does not clamp NPZ depth. Current default: 15.0. |
| `--max-fit-depth METRES` | Exclude deeper RealSense anchors from fitting. Current default: 30.0. |
| `--gt-max-depth METRES` | Mark generated output deeper than this as invalid (`NaN`), not clipped. Current default: 20.0. |
| `--d435-merge fill\|min` | Merge policy for projected D455 and D435 maps. Current default: `fill`. |
| `--splat` | Enable 2 x 2 projection splat. Current default. |
| `--no-splat` | Disable splat and preserve one projected target pixel per source projection. |
| `--fit-samples N` | Maximum near-weighted anchors used by fitting. Current default: 10000; minimum: 50. |
| `--near-sample-weight VALUE` | Sampling weight for anchors at 5 m or nearer, in both fitting and validation selection. Current default: 3.0. |
| `--ransac-validation-ratio RATIO` | Fraction of anchors held out from fitting for validation. Current default: 0.2; valid range `[0,1)`. |

### Sky segmentation

| Parser | Purpose |
|---|---|
| `--sky`, `--no-sky` | Enable or disable sky masking. |
| `--sky-param PATH` | NCNN `.param` model. |
| `--sky-bin PATH` | NCNN `.bin` weights. |
| `--sky-size N` | Square NCNN sky input size. Current config: 384. |
| `--sky-input-name NAME` | NCNN input blob. Current config: `in0`. |
| `--sky-output-name NAME` | NCNN output blob. Current config: `out0`. |
| `--sky-mean M0 M1 M2` | Three-channel preprocessing mean. |
| `--sky-norm N0 N1 N2` | Three-channel preprocessing normalization. |
| `--sky-no-sigmoid` | Treat model output as an existing probability. Current config behavior. |
| `--sky-sigmoid` | Apply sigmoid to the selected output. |
| `--sky-thresh VALUE` | Binary sky probability threshold. Current config: 0.5. |
| `--sky-invert`, `--no-sky-invert` | Invert or preserve the resulting sky mask. |
| `--sky-heuristic`, `--no-sky-heuristic` | Use/disable brightness-texture heuristic instead of NCNN. |
| `--sky-gpu`, `--no-sky-gpu` | Enable/disable NCNN GPU execution. |
| `--sky-refine`, `--no-sky-refine` | Enable/disable guided sky-mask refinement. |
| `--sky-refine-radius N` | Guided-filter radius. Current config: 24. |
| `--sky-refine-eps VALUE` | Guided-filter regularization epsilon. Current config: 0.001. |
| `--sky-refine-low VALUE` | Low-confidence refinement threshold. Current config: 0.3. |
| `--sky-refine-high VALUE` | High-confidence refinement threshold. Current config: 0.5. |
| `--sky-refine-bias VALUE` | Refinement bias parameter. Current config: 0.8. |
| `--sky-refine-bilateral`, `--sky-refine-no-bilateral` | Enable/disable bilateral smoothing during refinement. |

### Legacy QC

| Parser | Purpose |
|---|---|
| `--qc`, `--no-qc` | Enable or disable moving suspicious output to `_review`. |
| `--qc-subdir NAME` | Review subdirectory. Current default: `_review`. |
| `--qc-min-spread VALUE` | Minimum `depth_p90/depth_p10`. Current default: 1.5. |
| `--qc-min-s VALUE` | Minimum absolute affine scale. Current default: 0.005. |
| `--qc-min-inl RATIO` | Minimum fitting inlier ratio. Current default: 0.3. |

## Notes and limitations

- The exported dense depth is a DA-V2-based metric **pseudo-GT**, not a direct
  dense RealSense measurement outside RealSense coverage.
- One global scale and shift cannot correct object-specific/local DA-V2 errors.
- Projection errors are most visible near occlusion boundaries, thin branches,
  and unsynchronized motion.
- Fitting and validation both use near-weighted sampling; validation statistics
  therefore emphasize the near range by design.
- New validation metrics are logged for analysis but do not yet affect QC.
- Always use the calibration and GT config matching the recording date and
  target stereo pair.
