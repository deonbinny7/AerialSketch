"""
utils/helpers.py
================
Shared utility functions: FPS counter, glow effect, save helper,
colour utilities, and telemetry hooks.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  FPS Counter
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Exponential moving average FPS counter."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha    = alpha
        self._fps      = 0.0
        self._prev_t   = time.perf_counter()

    def tick(self) -> float:
        now      = time.perf_counter()
        inst_fps = 1.0 / max(now - self._prev_t, 1e-6)
        self._fps = self._alpha * inst_fps + (1 - self._alpha) * self._fps
        self._prev_t = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ─────────────────────────────────────────────────────────────────────────────
#  Glow effect
# ─────────────────────────────────────────────────────────────────────────────

def draw_glow(frame: np.ndarray, center: Tuple[int,int],
              radius: int, color: Tuple, alpha: float = 0.55) -> None:
    """
    Draw a soft radial glow around a point by layering blurred circles.
    Mutates `frame` in place.
    """
    overlay = frame.copy()
    for r_mult, a_mult in [(2.5, 0.15), (1.8, 0.30), (1.2, 0.55), (1.0, 0.90)]:
        cv2.circle(overlay, center, int(radius * r_mult), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Solid core dot
    cv2.circle(frame, center, radius // 3, (255, 255, 255), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def put_text(frame: np.ndarray, text: str, origin: Tuple[int,int],
             scale: float = 0.65, color: Tuple = (255, 255, 255),
             thickness: int = 2, shadow: bool = True) -> None:
    """Draw text with an optional drop-shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(frame, text,
                    (origin[0] + 1, origin[1] + 1),
                    font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(frame: np.ndarray, pt1: Tuple, pt2: Tuple,
                      color: Tuple, radius: int = 12,
                      thickness: int = -1, alpha: float = 1.0) -> None:
    """Draw a filled or outlined rounded rectangle. Supports transparency."""
    overlay = frame.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    # Four corner circles
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness, cv2.LINE_AA)
    # Fill rectangles
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, thickness)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        frame[:] = overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_drawing(canvas: np.ndarray, output_dir: str = "saved/") -> str:
    """Save canvas to a timestamped PNG. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"sketch_{ts}.png")
    cv2.imwrite(path, canvas)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Colour utilities
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_hex(bgr: Tuple) -> str:
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def darken(bgr: Tuple, factor: float = 0.6) -> Tuple:
    return tuple(int(c * factor) for c in bgr)


def lighten(bgr: Tuple, factor: float = 1.4) -> Tuple:
    return tuple(min(255, int(c * factor)) for c in bgr)
