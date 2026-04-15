"""
recognition/preprocess.py  v2.0 — Production Upgrade
======================================================
Converts a list of (x,y) stroke points into normalised images and feature
vectors for both rule-based and ML classifiers.

Two feature-extraction paths:
  extract_geometric_features  – fast, raw-point features (straightness, circularity)
  extract_contour_features    – renders stroke → advanced morphological cleanup → contour analysis

New functions in v2.0
---------------------
  interpolate_stroke()  – fills pixel gaps > N px (fixes hollow circles from fast movement)
  normalize_stroke()    – centers + scales points to a standard coordinate space

Upgrade notes vs v1.0
---------------------
[NEW]  interpolate_stroke      — gap fill before feature extraction (Upgrade #4)
[NEW]  normalize_stroke        — size-invariant representation (Upgrade #4)
[IMPROVE] Auto-close threshold — now pixel-based ratio of bbox diagonal (was 30% of canvas diag)
[IMPROVE] Minimum contour area — raised to 1200 px² (was 800) (Upgrade #3)
[IMPROVE] Multi-scale morph    — two-pass close for stronger gap bridging (Upgrade #2)
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from config import (
    RECOG_IMAGE_SIZE,
    STROKE_GAP_FILL_PX,
    CONTOUR_MIN_AREA,
    CLOSE_GAP_RATIO,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Stroke pre-processing utilities  (Upgrade #4)
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_stroke(points: List[Tuple[int, int]],
                       max_gap_px: int = STROKE_GAP_FILL_PX) -> List[Tuple[int, int]]:
    """
    Fill pixel-level gaps in a stroke caused by fast hand movement.

    Problem being solved
    --------------------
    When the user's hand moves quickly, consecutive (x,y) samples can be
    12-25 px apart, leaving visible gaps in rendered circles/squares that
    cause the contour to look broken and trigger incorrect shape detection.

    Approach
    --------
    For each consecutive pair of points, if the Euclidean distance exceeds
    `max_gap_px`, linearly interpolate as many intermediate points as needed
    to ensure no gap larger than max_gap_px remains.

    Parameters
    ----------
    points     : raw stroke points as (x, y) tuples
    max_gap_px : maximum allowed gap in pixels — default from config
    """
    if len(points) < 2:
        return points

    result: List[Tuple[int, int]] = [points[0]]
    for i in range(1, len(points)):
        p0 = np.array(points[i - 1], dtype=float)
        p1 = np.array(points[i],     dtype=float)
        dist = np.linalg.norm(p1 - p0)
        if dist > max_gap_px:
            n_steps = int(np.ceil(dist / max_gap_px))
            for step in range(1, n_steps):
                t  = step / n_steps
                ip = (int(round(p0[0] + t * (p1[0] - p0[0]))),
                      int(round(p0[1] + t * (p1[1] - p0[1]))))
                result.append(ip)
        result.append(points[i])
    return result


def normalize_stroke(points: List[Tuple[int, int]],
                     target_size: int = 200) -> List[Tuple[int, int]]:
    """
    Center and scale stroke points to fit within a target_size × target_size box.

    Purpose
    -------
    Removes size-variance from contour feature extraction. A tiny circle and a
    large circle should produce the same circularity score. Without normalisation,
    a small stroke produces a tiny contour that gets noisier measurements.

    Returns the transformed points. Does NOT mutate the input list.
    """
    if len(points) < 2:
        return points

    pts = np.array(points, dtype=float)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    w = x_max - x_min + 1e-6
    h = y_max - y_min + 1e-6

    scale = (target_size * 0.85) / max(w, h)   # leave 7.5% border each side
    offset = (target_size - w * scale) / 2, (target_size - h * scale) / 2

    normalized = [
        (int(round((x - x_min) * scale + offset[0])),
         int(round((y - y_min) * scale + offset[1])))
        for x, y in points
    ]
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
#  Stroke → normalised image  (unchanged API, improved internals)
# ─────────────────────────────────────────────────────────────────────────────

def stroke_to_image(points: List[Tuple[int, int]], size: int = RECOG_IMAGE_SIZE) -> np.ndarray:
    """
    Render a list of 2D points onto a white-on-black binary image of shape
    (size, size, 1).  The stroke is normalised to fill most of the canvas.

    Changes v2.0: applies interpolate_stroke before rendering to ensure
    no pixel gaps remain for the CNN to misinterpret as broken shapes.
    """
    if len(points) < 2:
        return np.zeros((size, size, 1), dtype=np.uint8)

    # Gap-fill before rendering
    points = interpolate_stroke(points)

    pts = np.array(points, dtype=np.int32)

    # Shift so bounding box starts at (0,0)
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


# ─────────────────────────────────────────────────────────────────────────────
#  Raw-point geometric features (used as lightweight fallback)
# ─────────────────────────────────────────────────────────────────────────────

def extract_geometric_features(points: List[Tuple[int, int]]) -> dict:
    """
    Compute hand-crafted features from raw points.

    Returns dict with keys: aspect_ratio, circularity, num_corners,
                             straightness, bbox_w, bbox_h, point_count
    """
    if len(points) < 3:
        return {}

    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    aspect_ratio = w / (h + 1e-6)

    perimeter = 0.0
    for i in range(1, len(pts)):
        perimeter += float(np.linalg.norm(pts[i] - pts[i-1]))

    area = 0.5 * abs(
        sum(pts[i][0] * pts[(i+1) % len(pts)][1]
            - pts[(i+1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts)))
    )
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    hull = cv2.convexHull(pts)
    epsilon = 0.04 * cv2.arcLength(hull, True)
    approx  = cv2.approxPolyDP(hull, epsilon, True)
    num_corners = len(approx)

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


# ─────────────────────────────────────────────────────────────────────────────
#  Contour-based features  (Upgrades #1 #2 #3 #8)
# ─────────────────────────────────────────────────────────────────────────────

def extract_contour_features(points: List[Tuple[int, int]]) -> dict:
    """
    Render the stroke onto an image, apply advanced morphological cleanup,
    auto-close the shape, and extract robust contour metrics.

    Upgrade summary vs v1.0
    -----------------------
    [IMPROVE] interpolate_stroke() called first — eliminates hollow-circle problem
    [IMPROVE] normalize_stroke()  called first — size-invariant measurements
    [IMPROVE] Auto-close based on bbox diagonal ratio (was canvas diagonal)
    [IMPROVE] Two-pass morph-close with different kernel sizes (small gaps + large gaps)
    [IMPROVE] Minimum area raised from 800 → CONTOUR_MIN_AREA (1200) px²
    [NEW]     contour_smoothness metric: ratio of hull perimeter to contour perimeter
    """
    if len(points) < 10:
        return {}

    # ── 0. Pre-process: fill gaps, then normalize size ────────────────────────
    pts_filled    = interpolate_stroke(points)
    pts_normed    = normalize_stroke(pts_filled, target_size=300)
    pts           = np.array(pts_normed, dtype=np.int32)

    # ── 1. Render to a padded canvas ─────────────────────────────────────────
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    w_box = max(x_max - x_min, 1)
    h_box = max(y_max - y_min, 1)

    PAD, SCALE = 30, 2
    canvas_w = w_box * SCALE + PAD * 2
    canvas_h = h_box * SCALE + PAD * 2
    img = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    scaled      = pts.astype(np.float32)
    scaled[:, 0] = (scaled[:, 0] - x_min) * SCALE + PAD
    scaled[:, 1] = (scaled[:, 1] - y_min) * SCALE + PAD
    scaled       = scaled.astype(np.int32)

    thickness = max(4, SCALE * 3)
    for i in range(1, len(scaled)):
        cv2.line(img, tuple(scaled[i-1]), tuple(scaled[i]), 255, thickness, cv2.LINE_AA)

    # ── 2. Improved Auto-close (bbox-diagonal based) ──────────────────────────
    start_pt, end_pt = scaled[0], scaled[-1]
    gap  = float(np.linalg.norm(end_pt.astype(float) - start_pt.astype(float)))
    # Use bounding box diagonal of the *rendered* stroke, not canvas diagonal
    # This gives stable closure decisions regardless of shape size.
    bbox_diag = float(np.sqrt(w_box ** 2 + h_box ** 2)) * SCALE + 1e-6
    closed    = (gap / bbox_diag) < CLOSE_GAP_RATIO   # default 0.25

    if closed:
        cv2.line(img, tuple(end_pt), tuple(start_pt), 255, thickness, cv2.LINE_AA)

    # ── 3. Advanced Preprocessing (Upgrades #2 + #3) ─────────────────────────
    # Gaussian blur to smooth pixelation
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # Two-pass morphological close:
    #   Pass 1 (small kernel): seals hair-thin gaps < 5px
    #   Pass 2 (large kernel): bridges medium gaps < 15px from fast movement
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_small, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_large, iterations=1)

    _, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)

    # ── 4. Find largest contour ───────────────────────────────────────────────
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < CONTOUR_MIN_AREA:   # raised from 800 → 1200
        return {}

    # ── 5. Core measurements ─────────────────────────────────────────────────
    peri         = cv2.arcLength(cnt, True)
    circularity  = (4 * np.pi * area) / (peri * peri + 1e-6)

    # Convex Hull — stabilises noisy contours (Upgrade #2)
    hull          = cv2.convexHull(cnt)
    hull_area     = cv2.contourArea(hull)
    hull_peri     = cv2.arcLength(hull, True)
    solidity      = area / (hull_area + 1e-6)

    # Contour smoothness: ratio of hull perimeter to raw contour perimeter
    # A perfect shape → smoothness ≈ 1.0; noisy jagged contour → closer to 0.5
    contour_smoothness = hull_peri / (peri + 1e-6)

    # Multi-tolerance vertex approximation (tight for counting, loose for stability)
    approx_3  = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    approx_6  = cv2.approxPolyDP(cnt, 0.06 * peri, True)
    nv_3, nv_6 = len(approx_3), len(approx_6)

    x_b, y_b, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = bw / (bh + 1e-6)

    # Straightness on original (non-normalized) points for more reliable line signal
    raw_pts   = np.array(points, dtype=np.float32)
    raw_peri  = float(sum(np.linalg.norm(raw_pts[i] - raw_pts[i-1]) for i in range(1, len(raw_pts))))
    raw_dist  = float(np.linalg.norm(raw_pts[-1] - raw_pts[0]))
    straightness = raw_dist / (raw_peri + 1e-6)

    return {
        "vertices_tight":       nv_3,
        "vertices_loose":       nv_6,
        "circularity":          round(circularity, 4),
        "solidity":             round(solidity, 4),
        "contour_smoothness":   round(contour_smoothness, 4),
        "aspect_ratio":         round(aspect_ratio, 4),
        "straightness":         round(straightness, 4),
        "area":                 area,
        "closed":               closed,
        "bbox_w":               bw,
        "bbox_h":               bh,
    }
