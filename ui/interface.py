"""
ui/interface.py
===============
Renders the complete UI onto each frame:
  * Dark top toolbar (color swatches, brush indicator, mode label, FPS)
  * Fingertip glow cursor
  * Recognition result badge (shape + confidence bar)
  * Keyboard shortcut legend
  * Full-screen toast notifications
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    TOOLBAR_HEIGHT, PALETTE, PALETTE_NAMES, BRUSH_SIZES,
    OVERLAY_ALPHA, SHOW_FPS, SHOW_GESTURE_LABEL,
    GLOW_RADIUS, CAMERA_WIDTH, CAMERA_HEIGHT,
)
from utils.helpers import put_text, draw_glow, draw_rounded_rect, darken


# ─────────────────────────────────────────────────────────────────────────────
#  Toast notification
# ─────────────────────────────────────────────────────────────────────────────

class Toast:
    """Transient overlaid message that fades after N seconds."""

    def __init__(self, text: str, duration: float = 2.5,
                 color: Tuple = (0, 200, 255)) -> None:
        self.text      = text
        self.color     = color
        self._expires  = time.monotonic() + duration

    @property
    def alive(self) -> bool:
        return time.monotonic() < self._expires

    @property
    def alpha(self) -> float:
        remaining = self._expires - time.monotonic()
        return max(0.0, min(1.0, remaining / 0.6))   # fade-out last 0.6s


# ─────────────────────────────────────────────────────────────────────────────
#  UIRenderer
# ─────────────────────────────────────────────────────────────────────────────

class UIRenderer:
    """Stateful renderer — call render() once per frame."""

    def __init__(self) -> None:
        self._last_shape:      Optional[str]   = None
        self._last_conf:       float           = 0.0
        self._badge_timer:     float           = 0.0
        self._badge_duration:  float           = 3.5   # seconds
        self._toast: Optional[Toast]           = None

    # ── Public API ───────────────────────────────────────────────────────────

    def set_detection(self, shape: str, confidence: float) -> None:
        """Called when a new shape is detected."""
        self._last_shape  = shape
        self._last_conf   = confidence
        self._badge_timer = time.monotonic()

    def show_toast(self, text: str, color: Tuple = (0, 200, 255)) -> None:
        """Show a transient full-width toast notification."""
        self._toast = Toast(text, color=color)

    def render(self,
               frame: np.ndarray,
               fps: float,
               gesture_label: str,
               color_idx: int,
               brush_idx: int,
               fingertip: Optional[Tuple[int,int]],
               is_drawing: bool,
               debug: bool = False) -> np.ndarray:
        """
        Compose all UI layers onto `frame` (mutates in place) and return it.
        """
        self._draw_toolbar(frame, fps, gesture_label, color_idx, brush_idx, debug)
        if fingertip:
            self._draw_cursor(frame, fingertip, color_idx, is_drawing)
        self._draw_badge(frame)
        self._draw_shortcut_legend(frame)
        if self._toast and self._toast.alive:
            self._draw_toast(frame, self._toast)
        elif self._toast:
            self._toast = None
        return frame

    # ── Private renderers ────────────────────────────────────────────────────

    def _draw_toolbar(self, frame, fps, gesture_label,
                      color_idx, brush_idx, debug):
        h, w = TOOLBAR_HEIGHT, CAMERA_WIDTH

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 8, 18), -1)
        cv2.addWeighted(overlay, 1 - OVERLAY_ALPHA, frame, OVERLAY_ALPHA, 0, frame)

        # Bottom border of toolbar
        cv2.line(frame, (0, h), (w, h), PALETTE[color_idx], 2)

        # ── Title ──────────────────────────────────────────────────────────
        put_text(frame, "AerialSketch", (16, 50), scale=1.0,
                 color=(220, 220, 255), thickness=2)

        # ── Color swatches ─────────────────────────────────────────────────
        swatch_start = 250
        swatch_size  = 28
        swatch_gap   = 38
        for i, col in enumerate(PALETTE):
            cx = swatch_start + i * swatch_gap
            cy = h // 2
            # Selection ring
            if i == color_idx:
                cv2.circle(frame, (cx, cy), swatch_size // 2 + 5,
                           (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), swatch_size // 2, col, -1, cv2.LINE_AA)
            put_text(frame, PALETTE_NAMES[i][0], (cx - 5, cy + 5),
                     scale=0.45, color=(10, 10, 10), thickness=1, shadow=False)

        # ── Brush indicator ────────────────────────────────────────────────
        bx = swatch_start + len(PALETTE) * swatch_gap + 30
        by = h // 2
        bs = BRUSH_SIZES[brush_idx]
        cv2.circle(frame, (bx, by), max(3, bs), PALETTE[color_idx], -1, cv2.LINE_AA)
        put_text(frame, f"B:{bs}px", (bx + 18, by + 6), scale=0.5,
                 color=(180, 180, 200))

        # ── Gesture mode label ─────────────────────────────────────────────
        if SHOW_GESTURE_LABEL:
            mode_col = {
                "DRAWING": (0, 255, 140),
                "PAUSED":  (0, 160, 255),
                "CLEAR":   (0,  80, 255),
                "IDLE":    (120, 120, 150),
            }.get(gesture_label, (180, 180, 180))
            put_text(frame, f"● {gesture_label}",
                     (w - 240, h // 2 + 8), scale=0.65, color=mode_col)

        # ── FPS ────────────────────────────────────────────────────────────
        if SHOW_FPS:
            fps_col = (0, 220, 80) if fps >= 25 else (0, 100, 255)
            put_text(frame, f"FPS: {fps:.0f}", (w - 130, 26),
                     scale=0.55, color=fps_col)

        # ── Debug extras ───────────────────────────────────────────────────
        if debug:
            put_text(frame, "DEBUG", (w - 80, h - 6), scale=0.4,
                     color=(255, 80, 80))

    def _draw_cursor(self, frame, pt, color_idx, is_drawing):
        col = PALETTE[color_idx]
        r   = GLOW_RADIUS
        if is_drawing:
            draw_glow(frame, pt, r, col, alpha=0.6)
        else:
            # Just a dim ring when not actively drawing
            cv2.circle(frame, pt, r, darken(col, 0.7), 1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, col, -1, cv2.LINE_AA)

    def _draw_badge(self, frame):
        """Show the shape detection result for a few seconds."""
        if self._last_shape is None:
            return
        elapsed = time.monotonic() - self._badge_timer
        if elapsed > self._badge_duration:
            return

        # Fade out in last 0.8s
        alpha = 1.0 if elapsed < (self._badge_duration - 0.8) else \
                max(0, (self._badge_duration - elapsed) / 0.8)

        h, w = frame.shape[:2]
        bw, bh   = 300, 80
        bx       = (w - bw) // 2
        by       = h - bh - 20

        # Background panel
        overlay = frame.copy()
        draw_rounded_rect(overlay, (bx, by), (bx + bw, by + bh),
                          (18, 18, 32), radius=14, alpha=1.0)
        cv2.addWeighted(overlay, alpha * 0.75,
                        frame, 1 - alpha * 0.75, 0, frame)

        # Shape name
        label = f"{self._last_shape.upper()}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = bx + (bw - tw) // 2
        put_text(frame, label, (tx, by + 38), scale=0.9,
                 color=(0, 200, 255), thickness=2)

        # Confidence bar
        bar_w    = int((bw - 40) * self._last_conf * alpha)
        bar_x    = bx + 20
        bar_y    = by + 52
        bar_h    = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bw - 40, bar_y + bar_h),
                      (40, 40, 60), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (0, 200, 255), -1, cv2.LINE_AA)
        put_text(frame, f"{int(self._last_conf*100)}%",
                 (bar_x + bw - 20, bar_y + 9),
                 scale=0.4, color=(200, 200, 220), thickness=1, shadow=False)

    def _draw_shortcut_legend(self, frame):
        h = frame.shape[0]
        shortcuts = [
            ("C", "Next color"),
            ("B", "Brush size"),
            ("Z", "Undo"),
            ("Y", "Redo"),
            ("S", "Save"),
            ("D", "Debug"),
            ("Q", "Quit"),
        ]
        y = h - 10
        x = 16
        for key, label in shortcuts:
            put_text(frame, f"[{key}]{label}", (x, y), scale=0.38,
                     color=(100, 100, 140), thickness=1, shadow=False)
            x += len(f"[{key}]{label}") * 8 + 6

    def _draw_toast(self, frame, toast: Toast):
        h, w = frame.shape[:2]
        text = toast.text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx = (w - tw) // 2
        ty = TOOLBAR_HEIGHT + 60

        overlay = frame.copy()
        pad = 14
        draw_rounded_rect(overlay,
                          (tx - pad, ty - th - pad),
                          (tx + tw + pad, ty + pad),
                          (18, 18, 32), radius=12, alpha=1.0)
        cv2.addWeighted(overlay, toast.alpha * 0.80,
                        frame, 1 - toast.alpha * 0.80, 0, frame)
        put_text(frame, text, (tx, ty), scale=0.80, color=toast.color)
