#!/usr/bin/env python3
"""Summarize metric-depth experiment folders into one sortable CSV."""

import argparse
import csv
import json
from pathlib import Path


METRIC_DEPTH_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = METRIC_DEPTH_DIR / "out_data" / "runs"


def load_config(path):
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except (ImportError, yaml.YAMLError) as exc:
            print(f"warning: cannot read {path}: {exc}")
            return {}


def load_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value, default=float("inf")):
    try:
        value = float(value)
        return value if value == value else default
    except (TypeError, ValueError):
        return default


def summarize(run_dir):
    cfg = load_config(run_dir / "config.yaml")
    epochs = load_csv(run_dir / "metrics.csv")
    tests = load_csv(run_dir / "test_metrics.csv")
    best = min(epochs, key=lambda row: number(row.get("val_abs_rel"))) if epochs else {}
    last = epochs[-1] if epochs else {}
    test = tests[-1] if tests else {}
    loss = cfg.get("loss", {})
    loss_name = loss.get("name") if isinstance(loss, dict) else loss

    row = {
        "run": run_dir.name,
        "status": "complete" if tests else ("training" if epochs else "not_started"),
        "encoder": cfg.get("encoder"),
        "model_description": cfg.get("model_description"),
        "loss": loss_name,
        "data": cfg.get("data"),
        "img_size": cfg.get("img_size"),
        "batch_size": cfg.get("bs"),
        "learning_rate": cfg.get("lr"),
        "planned_epochs": cfg.get("epochs"),
        "completed_epochs": len(epochs),
        "best_epoch": best.get("epoch"),
        "best_val_abs_rel": best.get("val_abs_rel"),
        "best_val_mae_m": best.get("val_mae"),
        "best_val_rmse_m": best.get("val_rmse"),
        "best_val_silog": best.get("val_silog"),
        "best_val_d1": best.get("val_d1"),
        "last_epoch": last.get("epoch"),
        "last_train_loss": last.get("train_total_loss"),
        "last_val_abs_rel": last.get("val_abs_rel"),
        "test_abs_rel": test.get("test_abs_rel"),
        "test_mae_m": test.get("test_mae"),
        "test_rmse_m": test.get("test_rmse"),
        "test_silog": test.get("test_silog"),
        "test_d1": test.get("test_d1"),
        "test_frames": test.get("test_evaluated_frames"),
        "best_checkpoint": (run_dir / "best.pt").is_file(),
        "last_checkpoint": (run_dir / "last.pt").is_file(),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--sort-by", default="best_val_abs_rel")
    args = parser.parse_args()

    # A run is one direct child of runs/. Do not count nested test/visualization
    # output folders that may also contain a resolved config.yaml.
    run_dirs = sorted(p.parent for p in args.runs_dir.glob("*/config.yaml"))
    rows = [summarize(path) for path in run_dirs]
    if not rows:
        raise SystemExit(f"No run containing config.yaml found under {args.runs_dir}")
    if args.sort_by not in rows[0]:
        raise SystemExit(f"Unknown --sort-by {args.sort_by!r}")
    rows.sort(key=lambda row: number(row.get(args.sort_by)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "runs_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'run':42} {'best epoch':>10} {'val abs_rel':>11} {'test MAE':>10} status")
    for row in rows:
        print(f"{row['run'][:42]:42} {str(row['best_epoch']):>10} "
              f"{str(row['best_val_abs_rel'])[:11]:>11} "
              f"{str(row['test_mae_m'])[:10]:>10} {row['status']}")
    print(f"Saved {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
