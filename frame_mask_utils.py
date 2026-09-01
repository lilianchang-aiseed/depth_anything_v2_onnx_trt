"""Shared fisheye-frame masking helpers for DA-V2 inference scripts."""

import argparse

import cv2
import numpy as np


def parse_bool(value):
    """Argparse converter that accepts explicit ``--frame-mask True/False``."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def build_frame_valid_mask(image: np.ndarray, threshold: int = 3,
                           erode_pixels: int = 3) -> np.ndarray:
    """Estimate the convex fisheye footprint while retaining dark pixels inside it."""
    nonblack = np.max(image, axis=2) > threshold
    y, x = np.nonzero(nonblack)
    if x.size < 3:
        raise ValueError("Cannot estimate frame mask: too few non-black pixels")
    hull = cv2.convexHull(np.column_stack((x, y)).astype(np.int32))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if erode_pixels > 0:
        size = erode_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.erode(mask, kernel)
    return mask.astype(bool)


def fill_invalid_with_nearest(image: np.ndarray,
                              valid_mask: np.ndarray) -> np.ndarray:
    """Fill invalid pixels with the nearest valid pixel using OpenCV labels."""
    invalid = (~valid_mask).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        invalid, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    y, x = np.nonzero(valid_mask)
    lookup = np.zeros((int(labels.max()) + 1, 2), dtype=np.int32)
    lookup[labels[y, x]] = np.column_stack((y, x))
    filled = image.copy()
    invalid_y, invalid_x = np.nonzero(~valid_mask)
    nearest = lookup[labels[invalid_y, invalid_x]]
    filled[invalid_y, invalid_x] = image[nearest[:, 0], nearest[:, 1]]
    return filled


def prepare_frame(image: np.ndarray):
    """Return nearest-filled input and its eroded valid mask."""
    valid_mask = build_frame_valid_mask(image)
    return fill_invalid_with_nearest(image, valid_mask), valid_mask

