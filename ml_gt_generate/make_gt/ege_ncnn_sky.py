"""Shared EGE NCNN sky-segmentation inference for GT generation tools."""

from pathlib import Path

import cv2
import numpy as np


def _bias(value, bias=0.8):
    return value / (((1.0 / bias) - 2.0) * (1.0 - value) + 1.0)


def refine_sky_probability(probability, gray, radius=24, eps=1e-3,
                           low=0.3, high=0.5, bias=0.8,
                           bilateral=True):
    """Confidence-weighted guided-filter refinement used by the GT pipeline."""
    probability = probability.astype(np.float32)
    guide = gray.astype(np.float32)
    if guide.max() > 1.5:
        guide /= 255.0
    confidence = np.full_like(probability, 0.01, dtype=np.float32)
    low_mask, high_mask = probability < low, probability > high
    confidence[low_mask] = np.maximum(
        _bias((low - probability[low_mask]) / low, bias), 0.01)
    confidence[high_mask] = np.maximum(
        _bias((probability[high_mask] - high) / (1.0 - high), bias), 0.01)
    kernel = (2 * int(radius) + 1,) * 2

    def box(value):
        return cv2.boxFilter(value, -1, kernel, normalize=False,
                             borderType=cv2.BORDER_REPLICATE)

    total = box(confidence) + 1e-8
    mean_guide = box(confidence * guide) / total
    mean_prob = box(confidence * probability) / total
    variance = box(confidence * guide * guide) / total - mean_guide**2
    covariance = box(confidence * guide * probability) / total
    covariance -= mean_guide * mean_prob
    slope = covariance / (variance + float(eps))
    offset = mean_prob - slope * mean_guide
    refined = box(confidence * slope) / total * guide
    refined += box(confidence * offset) / total
    refined = np.clip(refined, 0.0, 1.0)
    return cv2.bilateralFilter(refined, 0, 0.08, 8) if bilateral else refined


class EgeNcnnSkySegmenter:
    """Run EGE sky segmentation and return probability/mask at input size."""

    def __init__(self, param, weights, size=384, input_name="in0",
                 output_name="out5", mean=(117.790845,) * 3,
                 norm=(1.0 / 64.18484,) * 3, threshold=0.5,
                 invert=False, use_gpu=False, dynamic_input_scale=True,
                 refine=False, refine_radius=24, refine_eps=1e-3,
                 refine_low=0.3, refine_high=0.5, refine_bias=0.8,
                 refine_bilateral=True):
        import ncnn

        param = Path(param).expanduser().resolve()
        weights = Path(weights).expanduser().resolve()
        if not param.is_file() or not weights.is_file():
            raise FileNotFoundError(f"EGE model not found: {param}, {weights}")

        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = bool(use_gpu)
        if self.net.load_param(str(param)) != 0:
            raise RuntimeError(f"Failed to load NCNN param: {param}")
        if self.net.load_model(str(weights)) != 0:
            raise RuntimeError(f"Failed to load NCNN weights: {weights}")

        self.size = int(size)
        self.input_name = str(input_name)
        self.output_name = str(output_name)
        self.mean = list(mean)
        self.norm = list(norm)
        self.threshold = float(threshold)
        self.invert = bool(invert)
        self.dynamic_input_scale = bool(dynamic_input_scale)
        self.refine = bool(refine)
        self.refine_radius = int(refine_radius)
        self.refine_eps = float(refine_eps)
        self.refine_low = float(refine_low)
        self.refine_high = float(refine_high)
        self.refine_bias = float(refine_bias)
        self.refine_bilateral = bool(refine_bilateral)

    def probability(self, bgr):
        """Return an HxW float32 sky probability map."""
        if bgr is None or bgr.size == 0:
            raise ValueError("Empty image passed to EGE sky segmentation")
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        image = np.ascontiguousarray(bgr[..., :3])

        tensor = self.ncnn.Mat.from_pixels_resize(
            image, self.ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            w, h, self.size, self.size,
        )
        tensor.substract_mean_normalize(self.mean, self.norm)

        if self.dynamic_input_scale:
            values = np.asarray(tensor)
            low, high = float(values.min()), float(values.max())
            scale = 1.0 if high <= low else 255.0 / (high - low)
            tensor.substract_mean_normalize([low] * 3, [scale] * 3)

        extractor = self.net.create_extractor()
        extractor.set_light_mode(True)
        if extractor.input(self.input_name, tensor) != 0:
            raise RuntimeError(f"NCNN input blob not found: {self.input_name}")
        status, output = extractor.extract(self.output_name)
        if status != 0:
            raise RuntimeError(f"NCNN output blob not found: {self.output_name}")

        probability = np.asarray(output)
        if probability.ndim == 3:
            probability = probability[0] if probability.shape[0] == 1 else probability[-1]
        probability = cv2.resize(
            probability.astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        probability = np.clip(probability, 0.0, 1.0)
        return 1.0 - probability if self.invert else probability

    def mask(self, bgr, threshold=None):
        threshold = self.threshold if threshold is None else float(threshold)
        probability = self.probability(bgr)
        if self.refine:
            gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2GRAY)
            probability = refine_sky_probability(
                probability, gray, radius=self.refine_radius,
                eps=self.refine_eps, low=self.refine_low,
                high=self.refine_high, bias=self.refine_bias,
                bilateral=self.refine_bilateral,
            )
        return probability >= threshold
