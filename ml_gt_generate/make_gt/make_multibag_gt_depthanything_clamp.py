#!/usr/bin/env python3
"""Generate clamped GT for an ordered {bag: config} mapping in one process.

Edit BAG_CONFIGS below. Relative paths are resolved from this file's folder.
The public CLI intentionally has no --bag or --gt-config options.

Example:
  python3 make_gt/make_multibag_gt_depthanything_clamp.py \
    --out-dir out_data/multibag_gt --max-pairs 20 --frame-interval 50

Add --replace-old-bag to regenerate every BAG_CONFIGS job while preserving its
existing manifest bag number and output prefix.
"""

import argparse
import csv
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from make_gt_depthanything_clamp import (
    GTProcessor,
    output_filename_prefix,
    validate_output_prefix,
)


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_SHARE = Path(os.environ.get("COMMON_SHARE", "/COMMON_SHARE_NOT_SET"))

# Ordered mapping: exact MCAP path -> dated GT runtime config.
# Path("...") and strings containing $COMMON_SHARE are also accepted.
BAG_CONFIGS = {
    # 2026-08-25: prefer the recovered sixth recording over backup/original.
    Path("/home/share/bags/nx-2.0/0825/train_1_0-1/train_1_0_0.mcap"):
        SCRIPT_DIR / "gt_config_0825.yaml",
    Path("/home/share/bags/nx-2.0/0825/train_1_0-2/train_1_0_0.mcap"):
        SCRIPT_DIR / "gt_config_0825.yaml",
    Path("/home/share/bags/nx-2.0/0825/train_1_0-3/train_1_0_0.mcap"):
        SCRIPT_DIR / "gt_config_0825.yaml",
    Path("/home/share/bags/nx-2.0/0825/train_1_0-4/train_1_0_0.mcap"):
        SCRIPT_DIR / "gt_config_0825.yaml",
    Path("/home/share/bags/nx-2.0/0825/train_1_0-5/train_1_0_0.mcap"):
        SCRIPT_DIR / "gt_config_0825.yaml",
    Path("/home/share/bags/nx-2.0/0825/train_1_0-6_recovered/train_1_0_0.mcap"):
        SCRIPT_DIR / "gt_config_0825.yaml",

    # 2026-08-27: omit the explicitly named _original duplicate.
    Path("/home/share/bags/nx-2.0/0827/train_1_0_20260827_164533/train_1_0_20260827_164533_0.mcap"):
        SCRIPT_DIR / "gt_config_0827.yaml",
    Path("/home/share/bags/nx-2.0/0827/train_1_0_20260827_170239/train_1_0_20260827_170239_0.mcap"):
        SCRIPT_DIR / "gt_config_0827.yaml",

    # 2026-09-01 through 2026-09-09 use the 2_1 configuration.
    Path("/home/share/bags/nx-2.0/0901/train_2_1_20260901_155219/train_2_1_20260901_155219_0.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0901/train_2_1_20260901_155711/train_2_1_20260901_155711_0.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0901/train_2_1_20260901_161821_recovered/recovered.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0902/train_2_1_20260902_121639/train_2_1_20260902_121639_0.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0902/train_2_1_20260902_124546_recovered/recovered.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0904/train_2_1_20260904_155650/train_2_1_20260904_155650_0.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0904/train_2_1_20260904_160020/train_2_1_20260904_160020_0.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0907/train_2_1_20260907_181406/recovered.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0909/flight_data_2026_09_09-16_48_16/flight_data_2026_09_09-16_48_16_0.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",
    Path("/home/share/bags/nx-2.0/0909/flight_data_2026_09_09-16_50_58/flight_data_2026_09_09-16_50_58_0.mcap"):
        SCRIPT_DIR / "gt_config_0828.yaml",

    # 2026-09-16 and 2026-09-17: prefer newRect over original/recovered inputs.
    Path("/home/share/bags/nx-2.0/0916/flight_data_2026_09_16-15_23_09_newRect/flight_data_2026_09_16-15_23_09_newRect.mcap"):
        SCRIPT_DIR / "gt_config_0916.yaml",
    Path("/home/share/bags/nx-2.0/0916/flight_data_2026_09_16-15_30_04_newRect/flight_data_2026_09_16-15_30_04_newRect.mcap"):
        SCRIPT_DIR / "gt_config_0916.yaml",
    Path("/home/share/bags/nx-2.0/0916/flight_data_2026_09_16-15_37_41_newRect/flight_data_2026_09_16-15_37_41_newRect.mcap"):
        SCRIPT_DIR / "gt_config_0916.yaml",
    Path("/home/share/bags/nx-2.0/0917/flight_data_2026_09_17-15_31_25_newRect/flight_data_2026_09_17-15_31_25_newRect.mcap"):
        SCRIPT_DIR / "gt_config_0916.yaml",
    Path("/home/share/bags/nx-2.0/0917/flight_data_2026_09_17-16_43_56_newRect/flight_data_2026_09_17-16_43_56_newRect.mcap"):
        SCRIPT_DIR / "gt_config_0916.yaml",
    Path("/home/share/bags/nx-2.0/0917/flight_data_2026_09_17-16_49_03_newRect/flight_data_2026_09_17-16_49_03_newRect.mcap"):
        SCRIPT_DIR / "gt_config_0916.yaml",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def resolve_mapping_path(value):
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (expanded if expanded.is_absolute()
            else SCRIPT_DIR / expanded).resolve()


def manifest_bag_path(row):
    """Read both current and legacy bag-manifest layouts."""
    bag_path = row.get("bag_path", "")
    if Path(bag_path).suffix.lower() == ".mcap":
        return bag_path
    legacy_path = row.get("relative_path", "")
    if Path(legacy_path).suffix.lower() == ".mcap":
        return legacy_path
    return bag_path


def read_manifest(path):
    rows = {}
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[manifest_bag_path(row)] = row
    return rows


def append_manifest(path, bag_number, prefix, bag_path, config_path):
    create = not path.exists()
    if create:
        fieldnames = ("bag_number", "output_prefix", "bag_path", "gt_config")
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            fieldnames = tuple(next(csv.reader(stream)))
    if fieldnames == ("bag_number", "output_prefix", "relative_path", "bag_path"):
        row = {
            "bag_number": bag_number,
            "output_prefix": prefix,
            "relative_path": str(bag_path),
            "bag_path": str(bag_path),
        }
    elif fieldnames == ("bag_number", "output_prefix", "bag_path", "gt_config"):
        row = {
            "bag_number": bag_number,
            "output_prefix": prefix,
            "bag_path": str(bag_path),
            "gt_config": str(config_path),
        }
    else:
        raise RuntimeError(f"unsupported manifest columns: {fieldnames}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if create:
            writer.writeheader()
        writer.writerow(row)


def replace_targets(prefix, out_dir):
    """Return only this bag prefix's known generated files."""
    prefix = validate_output_prefix(prefix)
    layouts = (
        (out_dir / "dataset", f"{prefix}_*.npz"),
        (out_dir / "dataset" / "_review", f"{prefix}_*.npz"),
        (out_dir / "vis", f"{prefix}_*.png"),
        (out_dir / "vis" / "_review", f"{prefix}_*.png"),
    )
    targets = []
    for directory, pattern in layouts:
        if directory.is_dir():
            targets.extend(path for path in directory.glob(pattern))
    targets.extend((
        out_dir / "statistics" / f"{prefix}.csv",
        out_dir / "configs" / f"{prefix}_resolved_config.yaml",
        out_dir / "logs" / f"{prefix}.log",
        out_dir / "completed" / f"{prefix}.running",
        out_dir / "completed" / f"{prefix}.complete",
        out_dir / "completed" / f"{prefix}.failed",
    ))
    return sorted({path for path in targets if path.exists()})


def remove_old_bag_outputs(prefix, out_dir):
    """Remove a previously generated bag result without touching its manifest."""
    out_dir = out_dir.resolve()
    targets = replace_targets(prefix, out_dir)
    for path in targets:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"refusing to replace non-regular output: {path}")
        try:
            path.resolve().relative_to(out_dir)
        except ValueError as error:
            raise RuntimeError(
                f"refusing to replace output outside {out_dir}: {path}") from error

    if targets:
        print(f"[REPLACE] {prefix}: removing {len(targets)} old files")
        for path in targets:
            print(f"  remove {path}")
        for path in targets:
            path.unlink()
    else:
        print(f"[REPLACE] {prefix}: no old files found")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Process BAG_CONFIGS with one reusable DA-V2/EGE process",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--frame-interval", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--replace-old-bag",
        action="store_true",
        help=("regenerate mapped bags by removing only their existing "
              "prefix-matched outputs; keep bag numbers in bag_manifest.csv"),
    )
    args, forwarded = parser.parse_known_args(argv)
    blocked = {
        "--bag", "--gt-config", "--vis-dir", "--export-dir",
        "--stats-csv", "--resolved-config", "--output-prefix",
    }
    used_blocked = [item for item in forwarded if item.split("=", 1)[0] in blocked]
    if used_blocked:
        parser.error(f"multibag runner manages these options: {used_blocked}")
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    if args.frame_interval is not None and args.frame_interval < 1:
        parser.error("--frame-interval must be positive")
    return args, forwarded


def main(argv=None):
    args, forwarded = parse_args(argv)
    if not BAG_CONFIGS:
        raise SystemExit(
            f"BAG_CONFIGS is empty; edit {Path(__file__).resolve()} first"
        )

    jobs = [
        (resolve_mapping_path(bag), resolve_mapping_path(config))
        for bag, config in BAG_CONFIGS.items()
    ]
    for bag, config in jobs:
        if not bag.is_file() or bag.suffix.lower() != ".mcap":
            raise SystemExit(f"MCAP does not exist: {bag}")
        if not config.is_file():
            raise SystemExit(f"GT config does not exist: {config}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    dataset_dir = out_dir / "dataset"
    vis_dir = out_dir / "vis"
    statistics_dir = out_dir / "statistics"
    configs_dir = out_dir / "configs"
    logs_dir = out_dir / "logs"
    completed_dir = out_dir / "completed"
    manifest_path = out_dir / "bag_manifest.csv"

    print(f"Jobs: {len(jobs)}")
    print(f"Output: {out_dir}")
    if args.dry_run:
        for index, (bag, config) in enumerate(jobs, 1):
            print(f"[{index:02d}] {bag} -> {config}")
        return 0

    for directory in (
        dataset_dir, vis_dir, statistics_dir, configs_dir, logs_dir, completed_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    lock_path = out_dir / ".make_multibag.lock"
    lock_stream = lock_path.open("a+")
    try:
        import fcntl
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another multibag run is writing to {out_dir}")

    manifest = read_manifest(manifest_path)
    used_numbers = [int(row["bag_number"]) for row in manifest.values()]
    next_number = max(used_numbers, default=0) + 1
    processor = GTProcessor()
    success = failed = skipped = 0

    for bag, config in jobs:
        existing = manifest.get(str(bag))
        if existing:
            bag_number = int(existing["bag_number"])
            prefix = existing["output_prefix"]
        else:
            bag_number = next_number
            next_number += 1
            prefix = validate_output_prefix(
                f"{output_filename_prefix(bag)}_bag{bag_number:02d}"
            )
            append_manifest(manifest_path, bag_number, prefix, bag, config)
            manifest[str(bag)] = {
                "bag_number": str(bag_number), "output_prefix": prefix,
                "bag_path": str(bag), "gt_config": str(config),
            }

        complete_path = completed_dir / f"{prefix}.complete"
        running_path = completed_dir / f"{prefix}.running"
        log_path = logs_dir / f"{prefix}.log"
        if args.replace_old_bag:
            remove_old_bag_outputs(prefix, out_dir)
        if complete_path.exists():
            print(f"[SKIP] {prefix}: already complete")
            skipped += 1
            continue
        if running_path.exists() or log_path.exists():
            print(f"[FAIL] {prefix}: partial output/log already exists")
            failed += 1
            if args.stop_on_error:
                break
            continue

        job_argv = [
            "--bag", str(bag),
            "--gt-config", str(config),
            "--vis-dir", str(vis_dir),
            "--export-dir", str(dataset_dir),
            "--stats-csv", str(statistics_dir / f"{prefix}.csv"),
            "--resolved-config", str(
                configs_dir / f"{prefix}_resolved_config.yaml"),
            "--output-prefix", prefix,
            *forwarded,
        ]
        if args.max_pairs is not None:
            job_argv.extend(("--max-pairs", str(args.max_pairs)))
        if args.frame_interval is not None:
            job_argv.extend(("--frame-interval", str(args.frame_interval)))

        print(f"[RUN {bag_number:02d}] {bag.name}")
        running_path.write_text(f"bag={bag}\nconfig={config}\n", encoding="utf-8")
        try:
            with log_path.open("x", encoding="utf-8") as log:
                with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(
                    Tee(sys.stderr, log)
                ):
                    result = processor.process_bag(job_argv)
            running_path.replace(complete_path)
            with complete_path.open("a", encoding="utf-8") as stream:
                stream.write(f"result={result}\n")
            success += 1
        except (Exception, SystemExit) as error:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            print(f"[FAIL] {prefix}: {error}", file=sys.stderr)
            failed += 1
            if args.stop_on_error:
                break

    print(f"Summary: success={success} failed={failed} skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
