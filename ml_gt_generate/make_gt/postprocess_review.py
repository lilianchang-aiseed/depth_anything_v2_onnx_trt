#!/usr/bin/env python3
"""Move low-quality GT NPZ/PNG pairs into _review without deleting data.

By default this is a dry run. Add --apply after inspecting the printed list.

Usage:
dry-run: 
```bash
cd ~/Projects/da_V2

OUT_DIR="ml_gt_generate/out_data/YOUR_BATCH_OUTPUT"

python3 ml_gt_generate/make_gt/postprocess_review.py \
  --out-dir "$OUT_DIR" \
  --mae-0-2-max 1 \
  --mae-2-5-max 2 \
  --mae-rule both \
  --dirty-before 0827_train_1_0_20260827_164533_bag01=140 \
  --dirty-before 0827_train_1_0_20260827_170239_bag02=174 \
  --dirty-before 0901_train_2_1_20260901_155219_bag07=15
```

apply: 
```bash
python3 ml_gt_generate/make_gt/postprocess_review.py \
  --out-dir "$OUT_DIR" \
  --mae-0-2-max 1 \
  --mae-2-5-max 2 \
  --mae-rule both \
  --dirty-before 0827_train_1_0_20260827_164533_bag01=140 \
  --dirty-before 0827_train_1_0_20260827_170239_bag02=174 \
  --dirty-before 0901_train_2_1_20260901_155219_bag07=15 \
  --apply
```

"""
import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

try:
    from natsort import natsorted
except ImportError as error:
    raise SystemExit("natsort is required: pip install natsort") from error


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_dirty_rule(text):
    try:
        prefix, limit = text.rsplit("=", 1)
        limit = int(limit)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected PREFIX=FIRST_CLEAN_FRAME, for example session_bag01=140"
        ) from error
    if not prefix or limit < 1:
        raise argparse.ArgumentTypeError("prefix must be non-empty and frame >= 1")
    return prefix, limit


def main():
    parser = argparse.ArgumentParser(
        description="Move rejected dataset/visualization pairs into _review")
    parser.add_argument("--out-dir", required=True,
                        help="batch root containing dataset, vis, and statistics")
    parser.add_argument("--mae-0-2-max", type=float, default=1.0)
    parser.add_argument("--mae-2-5-max", type=float, default=2.0)
    parser.add_argument("--mae-rule", choices=("both", "either"), default="both",
                        help="both: require both MAEs to exceed limits (default)")
    parser.add_argument(
        "--dirty-before", action="append", type=parse_dirty_rule, default=[],
        metavar="PREFIX=FIRST_CLEAN_FRAME",
        help="review frames below the limit; may be supplied multiple times")
    parser.add_argument("--apply", action="store_true",
                        help="perform moves; without this flag only preview")
    args = parser.parse_args()

    root = Path(args.out_dir).expanduser().resolve()
    dataset_dir, vis_dir = root / "dataset", root / "vis"
    statistics_dir = root / "statistics"
    if not dataset_dir.is_dir() or not vis_dir.is_dir() or not statistics_dir.is_dir():
        parser.error("--out-dir must contain dataset/, vis/, and statistics/")
    csv_files = natsorted(statistics_dir.glob("*.csv"), key=lambda path: path.name)
    if not csv_files:
        parser.error(f"no per-bag CSV files found in {statistics_dir}")

    dirty_limits = dict(args.dirty_before)
    decisions = {}
    for csv_path in csv_files:
        prefix = csv_path.stem
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    frame = int(row["frame"])
                except (KeyError, TypeError, ValueError):
                    raise SystemExit(f"invalid frame column in {csv_path}")
                reasons = []
                if prefix in dirty_limits and frame < dirty_limits[prefix]:
                    reasons.append(f"manual_dirty_before_{dirty_limits[prefix]}")

                mae02 = finite_float(row.get("val_MAE_0_2m"))
                mae25 = finite_float(row.get("val_MAE_2_5m"))
                failed02 = mae02 is not None and mae02 > args.mae_0_2_max
                failed25 = mae25 is not None and mae25 > args.mae_2_5_max
                failed_mae = ((failed02 and failed25) if args.mae_rule == "both"
                              else (failed02 or failed25))
                if failed_mae:
                    reasons.append(
                        f"mae_0_2={mae02:.3f},mae_2_5={mae25:.3f}")
                if reasons:
                    decisions[f"{prefix}_{frame}"] = reasons

    for prefix in dirty_limits:
        if not any(path.stem == prefix for path in csv_files):
            raise SystemExit(f"dirty prefix has no matching statistics CSV: {prefix}")

    rows = []
    move_plans = []
    for stem, reasons in natsorted(decisions.items(), key=lambda item: item[0]):
        files = [(dataset_dir / f"{stem}.npz", dataset_dir / "_review" / f"{stem}.npz"),
                 (vis_dir / f"{stem}.png", vis_dir / "_review" / f"{stem}.png")]
        existing = [(source, target) for source, target in files if source.is_file()]
        already = [target for _, target in files if target.is_file()]
        if existing and already:
            raise SystemExit(f"mixed normal/review state for {stem}; refusing partial move")
        if len(existing) not in (0, len(files)) or len(already) not in (0, len(files)):
            raise SystemExit(f"NPZ/PNG pair is incomplete for {stem}; refusing partial move")
        if any(target.exists() for _, target in existing):
            raise SystemExit(f"review destination already exists for {stem}")
        status = "already_review" if already else "missing" if not existing else "move"
        rows.append({"sample": stem, "status": status, "reason": ";".join(reasons)})
        if existing:
            move_plans.append((stem, existing, reasons))

    for stem, files, reasons in move_plans:
        print(f"[{'MOVE' if args.apply else 'DRY-RUN'}] {stem}: {'; '.join(reasons)}")
        if args.apply:
            for source, target in files:
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)

    if args.apply:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest = root / f"postprocess_review_{stamp}.csv"
        with manifest.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sample", "status", "reason"))
            writer.writeheader(); writer.writerows(rows)
        print(f"manifest: {manifest}")
    print(f"summary: selected={len(decisions)} movable={len(move_plans)} "
          f"already_review={sum(row['status'] == 'already_review' for row in rows)} "
          f"mode={'apply' if args.apply else 'dry-run'}")


if __name__ == "__main__":
    main()
