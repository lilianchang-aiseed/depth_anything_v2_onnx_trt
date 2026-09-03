"""Metric-depth dataset over the .npz samples from make_gt_depthanything.py.

Target is `depth_aligned` == 1/(s*DA_Small + t): DA-V2-Small relative disparity
scaled by the per-frame RealSense affine. Regressing it distils (DA-Small +
affine) into a single metric net -> monocular deploy, no RealSense, no SML.

Returns image (3,h,w normalized RGB), depth (h,w metres), valid_mask (h,w bool).

Augmentation (train mode only; ported from train_sml_global.py):
  * horizontal flip (p=0.5)                 -- image + depth together
  * brightness   *= U(1-b, 1+b)
  * contrast      : mid + (x-mid)*U(1-c,1+c)
  * gamma         : x**U(1-g,1+g)           (off by default)
  * gaussian noise: +N(0, std), with prob p
Val mode is fully deterministic (no flip / photometric / random-crop).
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
                 sky_as_far=False,
                 augment=None,
                 aug_hflip=True, aug_brightness=0.3, aug_contrast=0.2,
                 aug_gamma=0.0, aug_noise_std=0.0118, aug_noise_p=0.3):
        assert mode in ("train", "val")
        self.mode = mode
        self.size = size
        self.target_key = target_key
        self.fallback_key = fallback_key
        self.use_stored_valid = use_stored_valid
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.sky_as_far = bool(sky_as_far)

        self.augment = (mode == "train") if augment is None else bool(augment)
        self.aug_hflip = bool(aug_hflip)
        self.aug_brightness = float(aug_brightness)
        self.aug_contrast = float(aug_contrast)
        self.aug_gamma = float(aug_gamma)
        self.aug_noise_std = float(aug_noise_std)
        self.aug_noise_p = float(aug_noise_p)

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

    def _photometric(self, img, rng):
        if self.aug_brightness > 0:
            img = img * rng.uniform(1 - self.aug_brightness, 1 + self.aug_brightness)
        if self.aug_contrast > 0:
            img = 0.5 + (img - 0.5) * rng.uniform(1 - self.aug_contrast, 1 + self.aug_contrast)
        if self.aug_gamma > 0:
            img = np.clip(img, 0, 1) ** rng.uniform(1 - self.aug_gamma, 1 + self.aug_gamma)
        if self.aug_noise_std > 0 and rng.random() < self.aug_noise_p:
            img = img + rng.normal(0, self.aug_noise_std, img.shape).astype(np.float32)
        return np.clip(img, 0.0, 1.0).astype(np.float32)

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

        # ---- augmentation (train only); NaN in depth carries invalidity through ----
        if self.augment:
            rng = np.random.default_rng()
            if self.aug_hflip and rng.random() < 0.5:
                image = image[:, ::-1].copy()
                depth = depth[:, ::-1].copy()
            image = self._photometric(image, rng)

        sample = self.transform({"image": image, "depth": depth})
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = (torch.isnan(sample["depth"]) == 0)
        sample["depth"][sample["valid_mask"] == 0] = 0
        sample["image_path"] = path
        return sample