#!/usr/bin/env python3
"""Extract ``left`` images from NPZ files.

Example:
  python3 sml/tools/npz_2_img_folder.py \
    --img-path /path/to/train_1_0-5_dataset \
    --out-dir sml/train_1_0-5_dataset

Output:
  sml/train_1_0-5_dataset/sample_0000.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", "--img_path", required=True,
                        help="Folder containing NPZ files")
    parser.add_argument("--out-dir", required=True,
                        help="Exact output directory")
    args = parser.parse_args()

    input_dir = Path(args.img_path).expanduser()
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No NPZ files in: {input_dir}")

    output_dir = Path(args.out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "left" not in data.files:
                print(f"Skip {path.name}: no 'left' field")
                continue
            image = data["left"]
        output = output_dir / f"{path.stem}.jpg"
        if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            print(f"Failed to save: {output}")
            continue
        saved += 1

    print(f"Saved {saved}/{len(files)} images to {output_dir}")


if __name__ == "__main__":
    main()
