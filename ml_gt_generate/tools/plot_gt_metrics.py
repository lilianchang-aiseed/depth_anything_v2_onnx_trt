#!/usr/bin/env python3
"""Plot every numeric GT statistic, with one consistent color per ROS bag."""
import argparse
import csv
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ml-gt")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from natsort import natsorted
except ImportError:
    def natsorted(items, key=None):
        def natural_key(item):
            value = str(key(item) if key else item)
            return [int(part) if part.isdigit() else part.lower()
                    for part in re.split(r"(\d+)", value)]
        return sorted(items, key=natural_key)


IDENTIFIERS = {"frame", "source_frame", "stamp"}
THRESHOLDS = {"val_MAE_0_2m": 1.0, "val_MAE_2_5m": 2.0}


def read_bag_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "frame" not in rows[0]:
        return None
    return rows


def numeric_value(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def main():
    parser = argparse.ArgumentParser(
        description="Generate one line plot per numeric GT statistics column")
    parser.add_argument("--input-dir", required=True,
                        help="batch output root or its statistics directory")
    parser.add_argument("--out-dir", default=None,
                        help="default: INPUT_DIR/metric_figures")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    search_dir = (input_dir / "statistics"
                  if (input_dir / "statistics").is_dir() else input_dir)
    csv_paths = natsorted(search_dir.glob("*.csv"), key=lambda path: path.name)
    bags = []
    for path in csv_paths:
        rows = read_bag_csv(path)
        if rows is not None:
            bags.append((path.stem, rows))
    if not bags:
        parser.error(f"no per-bag statistics CSV found in {search_dir}")

    columns = []
    for _, rows in bags:
        for name in rows[0]:
            if name in IDENTIFIERS or name in columns:
                continue
            if any(math.isfinite(numeric_value(row.get(name))) for row in rows):
                columns.append(name)

    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else input_dir / "metric_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab20").colors

    for metric in columns:
        figure, axis = plt.subplots(figsize=(13, 6))
        plotted = False
        for index, (bag, rows) in enumerate(bags):
            x = [int(float(row["frame"])) for row in rows]
            y = [numeric_value(row.get(metric)) for row in rows]
            if not any(math.isfinite(value) for value in y):
                continue
            axis.plot(x, y, color=colors[index % len(colors)], linewidth=1.2,
                      alpha=0.9, label=bag)
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        if metric in THRESHOLDS:
            threshold = THRESHOLDS[metric]
            axis.axhline(threshold, color="black", linestyle="--", linewidth=1,
                         label=f"review threshold = {threshold:g}")
        axis.set(title=metric, xlabel="Frame", ylabel=metric)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        figure.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)
        figure.savefig(out_dir / f"{safe_name}.png", dpi=args.dpi)
        plt.close(figure)

    print(f"bags={len(bags)} metrics={len(columns)} output={out_dir}")


if __name__ == "__main__":
    main()
