"""
recognition/preprocess.py
==========================
Converts a list of (x,y) stroke points into a normalised 2D image
suitable for both rule-based and ML-based classifiers.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from config import RECOG_IMAGE_SIZE


def stroke_to_image(points: List[Tuple[int,int]], size: int = RECOG_IMAGE_SIZE) -> np.ndarray:
    """
    Render a list of 2D points onto a white-on-black binary image of shape
    (size, size, 1).  The stroke is normalised to fill most of the canvas.

    Steps
    -----
    1. Draw the stroke on a temporary large canvas.
    2. Crop to the bounding box of the stroke.
    3. Pad to square.
    4. Resize to (size, size).
    5. Binarise.
    """
    if len(points) < 2:
        return np.zeros((size, size, 1), dtype=np.uint8)

    pts = np.array(points, dtype=np.int32)

    # Shift so that bounding box starts at (0,0)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    pts[:, 0] -= x_min
    pts[:, 1] -= y_min

    raw_w = int(pts[:, 0].max()) + 1
    raw_h = int(pts[:, 1].max()) + 1

    # Draw on oversized canvas (avoid aliasing at small sizes)
    scale = 4
    tmp = np.zeros((raw_h * scale, raw_w * scale), dtype=np.uint8)
    scaled = pts.copy()
    scaled[:, 0] *= scale
    scaled[:, 1] *= scale

    for i in range(1, len(scaled)):
        cv2.line(tmp, tuple(scaled[i-1]), tuple(scaled[i]), 255, 3 * scale, cv2.LINE_AA)

    # Pad to square
    h, w = tmp.shape
    side = max(h, w)
    pad_top  = (side - h) // 2
    pad_left = (side - w) // 2
    square = np.zeros((side, side), dtype=np.uint8)
    square[pad_top:pad_top+h, pad_left:pad_left+w] = tmp

    # Resize to target
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

    # Binarise
    _, binary = cv2.threshold(resized, 50, 255, cv2.THRESH_BINARY)

    return binary[:, :, np.newaxis]   # (size, size, 1)


def extract_geometric_features(points: List[Tuple[int,int]]) -> dict:
    """
    Compute hand-crafted features used by the rule-based recogniser.

    Returns
    -------
    dict with keys: aspect_ratio, circularity, num_corners, straightness,
                    bbox_w, bbox_h, point_count
    """
    if len(points) < 3:
        return {}

    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    # –– Aspect ratio ──────────────────────────────────────────────────────
    aspect_ratio = w / (h + 1e-6)

    # –– Perimeter & area for circularity ──────────────────────────────────
    perimeter = 0.0
    for i in range(1, len(pts)):
        perimeter += float(np.linalg.norm(pts[i] - pts[i-1]))
    # Approximate enclosed area via the shoelace formula
    area = 0.5 * abs(
        sum(pts[i][0] * pts[(i+1) % len(pts)][1]
            - pts[(i+1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts)))
    )
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    # –– Convex hull corner estimation ─────────────────────────────────────
    hull = cv2.convexHull(pts)
    # Approximate polygon to find corner count
    epsilon = 0.04 * cv2.arcLength(hull, True)
    approx  = cv2.approxPolyDP(hull, epsilon, True)
    num_corners = len(approx)

    # –– Straightness (1.0 = perfectly straight line) ──────────────────────
    if len(pts) >= 2:
        vec  = (pts[-1] - pts[0]).astype(float)
        dist = float(np.linalg.norm(vec)) + 1e-6
        straightness = dist / (perimeter + 1e-6)
    else:
        straightness = 0.0

    return {
        "aspect_ratio":  aspect_ratio,
        "circularity":   circularity,
        "num_corners":   num_corners,
        "straightness":  straightness,
        "bbox_w":        w,
        "bbox_h":        h,
        "point_count":   len(points),
    }
