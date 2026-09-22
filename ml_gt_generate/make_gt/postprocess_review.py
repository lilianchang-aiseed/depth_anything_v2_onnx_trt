#!/usr/bin/env python3
"""Move low-quality GT NPZ/PNG pairs into _review without deleting data.

By default this is a dry run. Add --apply after inspecting the printed list.
MAE filtering is disabled unless at least one --mae-*-max option is supplied.

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
  --dirty-interval ml_gt_generate/make_gt/dirty_data_nx2.0_5-affine.yaml
```

apply: 
```bash
python3 ml_gt_generate/make_gt/postprocess_review.py \
  --out-dir "$OUT_DIR" \
  --mae-0-2-max 1 \
  --mae-2-5-max 2 \
  --mae-rule both \
  --dirty-interval ml_gt_generate/make_gt/dirty_data_nx2.0_5-affine.yaml \
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

try:
    import yaml
except ImportError as error:
    raise SystemExit("PyYAML is required: pip install pyyaml") from error


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_dirty_intervals(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"dirty-interval YAML does not exist: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid dirty-interval YAML {path}: {error}") from error
    if document is None:
        return {}
    if not isinstance(document, dict) or set(document) != {"dirty_intervals"}:
        raise ValueError(
            "dirty-interval YAML must contain exactly one mapping named "
            "dirty_intervals")
    rules = document["dirty_intervals"]
    if not isinstance(rules, dict):
        raise ValueError("dirty_intervals must map each prefix to a list of ranges")

    normalized = {}
    for prefix, intervals in rules.items():
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("dirty-interval prefix must be a non-empty string")
        if not isinstance(intervals, list):
            raise ValueError(f"dirty intervals for {prefix!r} must be a list")
        normalized_intervals = []
        for interval in intervals:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(
                    f"dirty interval for {prefix!r} must be [FIRST_FRAME, LAST_FRAME]")
            first, last = interval
            if (isinstance(first, bool) or isinstance(last, bool)
                    or not isinstance(first, int) or not isinstance(last, int)
                    or first < 1 or last < first):
                raise ValueError(
                    f"dirty interval for {prefix!r} must satisfy "
                    "1 <= FIRST_FRAME <= LAST_FRAME")
            normalized_intervals.append((first, last))
        normalized_intervals.sort()
        for previous, current in zip(normalized_intervals,
                                     normalized_intervals[1:]):
            if current[0] <= previous[1]:
                raise ValueError(
                    f"dirty intervals overlap for {prefix!r}: "
                    f"{previous} and {current}")
        normalized[prefix] = normalized_intervals
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Move rejected dataset/visualization pairs into _review")
    parser.add_argument("--out-dir", required=True,
                        help="batch root containing dataset, vis, and statistics")
    parser.add_argument(
        "--mae-0-2-max", type=float,
        help="optional 0-2 m MAE limit; omitted means this range is not checked")
    parser.add_argument(
        "--mae-2-5-max", type=float,
        help="optional 2-5 m MAE limit; omitted means this range is not checked")
    parser.add_argument(
        "--mae-rule", choices=("both", "either"), default="both",
        help="combine only the MAE limits supplied on the command line")
    parser.add_argument(
        "--dirty-interval", type=Path, metavar="YAML",
        help="YAML file mapping prefixes to inclusive dirty frame intervals")
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

    try:
        dirty_intervals = (load_dirty_intervals(args.dirty_interval)
                           if args.dirty_interval else {})
    except ValueError as error:
        parser.error(str(error))
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
                for first, last in dirty_intervals.get(prefix, ()):
                    if first <= frame <= last:
                        reasons.append(f"manual_dirty_interval_{first}_{last}")
                        break

                mae_checks = []
                if args.mae_0_2_max is not None:
                    mae_checks.append((
                        "mae_0_2", finite_float(row.get("val_MAE_0_2m")),
                        args.mae_0_2_max))
                if args.mae_2_5_max is not None:
                    mae_checks.append((
                        "mae_2_5", finite_float(row.get("val_MAE_2_5m")),
                        args.mae_2_5_max))
                mae_failures = [
                    (name, value, limit) for name, value, limit in mae_checks
                    if value is not None and value > limit
                ]
                failed_mae = bool(mae_checks) and (
                    len(mae_failures) == len(mae_checks)
                    if args.mae_rule == "both" else bool(mae_failures))
                if failed_mae:
                    reasons.append(",".join(
                        f"{name}={value:.3f}>{limit:g}"
                        for name, value, limit in mae_failures))
                if reasons:
                    decisions[f"{prefix}_{frame}"] = reasons

    for prefix in dirty_intervals:
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
