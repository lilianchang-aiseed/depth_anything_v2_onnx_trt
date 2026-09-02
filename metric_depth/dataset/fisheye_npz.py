"""Metric-depth dataset over the .npz samples from make_gt_depthanything.py.

Target is `depth_aligned`  ==  1 / (s * DA_Small + t):  Depth-Anything-V2-Small
relative disparity scaled by the per-frame RealSense affine.  Training a metric
DA-V2 to regress it distils (DA-Small + affine) into a single metric net, so the
deploy path is one forward pass with no RealSense and no SML.

Returns exactly what train_fisheye.py / train.py expect:
    image      (3,h,w) float32, ImageNet-normalized RGB
    depth      (h,w)   float32, metres
    valid_mask (h,w)   bool
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class FisheyeNPZ(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518),
                 target_key="depth_aligned", fallback_key="rs_depth_L",
                 use_stored_valid=True, min_depth=0.2, max_depth=20.0,
                 sky_as_far=False):
        assert mode in ("train", "val")
        self.mode = mode
        self.size = size
        self.target_key = target_key
        self.fallback_key = fallback_key
        self.use_stored_valid = use_stored_valid
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.sky_as_far = bool(sky_as_far)

        with open(filelist_path, "r") as f:
            self.filelist = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w, height=net_h,
                resize_target=(mode == "train"),
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if mode == "train" else []))

    def __len__(self):
        return len(self.filelist)

    def _load_image(self, left):
        if left.ndim == 2:
            bgr = cv2.cvtColor(left.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            bgr = np.ascontiguousarray(left[..., :3]).astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, item):
        path = self.filelist[item]
        z = np.load(path, allow_pickle=False)

        image = self._load_image(z["left"])

        key = self.target_key if self.target_key in z.files else self.fallback_key
        depth = np.asarray(z[key], dtype=np.float32).copy()

        invalid = ~np.isfinite(depth) | (depth <= 0)
        invalid |= (depth < self.min_depth) | (depth > self.max_depth)
        if self.use_stored_valid and key == "depth_aligned" and "valid_mask" in z.files:
            invalid |= ~z["valid_mask"].astype(bool)

        if self.sky_as_far and "sky_mask" in z.files:
            sky = z["sky_mask"].astype(bool)
            depth[sky] = self.max_depth
            invalid[sky] = False

        depth[invalid] = np.nan   # NaN -> valid_mask, mirroring the Hypersim loader

        sample = self.transform({"image": image, "depth": depth})
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = (torch.isnan(sample["depth"]) == 0)
        sample["depth"][sample["valid_mask"] == 0] = 0
        sample["image_path"] = path
        return sample