"""
drawing/canvas.py
=================
Persistent drawing canvas with stroke smoothing, undo/redo,
multi-color support, and overlay blending onto the webcam feed.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, TOOLBAR_HEIGHT,
    CANVAS_BG_COLOR, PALETTE, BRUSH_SIZES,
    DEFAULT_COLOR_IDX, DEFAULT_BRUSH_IDX,
    SMOOTHING_WINDOW, MIN_STROKE_POINTS, MAX_STROKE_POINTS,
    UNDO_HISTORY_LIMIT,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Stroke — one continuous pencil-down motion
# ─────────────────────────────────────────────────────────────────────────────

class Stroke:
    """Stores a single drawn stroke with color & size metadata."""

    def __init__(self, color: Tuple, brush_size: int) -> None:
        self.color = color
        self.brush_size = brush_size
        self.points: List[Tuple[int,int]] = []

    def add_point(self, pt: Tuple[int,int]) -> None:
        self.points.append(pt)

    def is_valid(self) -> bool:
        return len(self.points) >= MIN_STROKE_POINTS

    def decimated(self, target: int = MAX_STROKE_POINTS) -> List[Tuple[int,int]]:
        """Return a uniformly-decimated copy if the stroke is too long."""
        if len(self.points) <= target:
            return self.points
        indices = np.linspace(0, len(self.points) - 1, target, dtype=int)
        return [self.points[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
#  MovingAverageSmoother
# ─────────────────────────────────────────────────────────────────────────────

class MovingAverageSmoother:
    """Smooths incoming (x, y) points with a sliding window average."""

    def __init__(self, window: int = SMOOTHING_WINDOW) -> None:
        self._buf: deque = deque(maxlen=window)

    def smooth(self, pt: Tuple[int,int]) -> Tuple[int,int]:
        self._buf.append(pt)
        avg_x = int(sum(p[0] for p in self._buf) / len(self._buf))
        avg_y = int(sum(p[1] for p in self._buf) / len(self._buf))
        return (avg_x, avg_y)

    def reset(self) -> None:
        self._buf.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  DrawingCanvas
# ─────────────────────────────────────────────────────────────────────────────

class DrawingCanvas:
    """
    Manages:
    * Persistent NumPy canvas layers
    * Active stroke accumulation
    * Undo / Redo stack
    * Color & brush cycling
    * Overlay merging with the webcam frame
    """

    def __init__(self, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT) -> None:
        self.width  = width
        self.height = height
        self._draw_height = height - TOOLBAR_HEIGHT  # usable area below toolbar

        # Canvas layers
        self._canvas: np.ndarray = self._blank_canvas()   # accumulated strokes
        self._overlay: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

        # Stroke tracking
        self._active_stroke: Optional[Stroke] = None
        self._completed_strokes: List[Stroke] = []
        self._smoother = MovingAverageSmoother()

        # Undo / Redo (stores canvas snapshots)
        self._undo_stack: deque = deque(maxlen=UNDO_HISTORY_LIMIT)
        self._redo_stack: deque = deque(maxlen=UNDO_HISTORY_LIMIT)

        # Appearance state
        self._color_idx = DEFAULT_COLOR_IDX
        self._brush_idx = DEFAULT_BRUSH_IDX
        self._eraser_mode = False

        # Last known fingertip for interpolation
        self._last_pt: Optional[Tuple[int,int]] = None
        self._is_drawing = False

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def color(self) -> Tuple:
        return PALETTE[self._color_idx]

    @property
    def brush_size(self) -> int:
        return BRUSH_SIZES[self._brush_idx]

    @property
    def canvas(self) -> np.ndarray:
        return self._canvas.copy()

    @property
    def active_stroke(self) -> Optional[Stroke]:
        return self._active_stroke

    @property
    def completed_strokes(self) -> List[Stroke]:
        return self._completed_strokes

    # ── Drawing API ─────────────────────────────────────────────────────────

    def begin_stroke(self) -> None:
        """Called when the user enters drawing mode."""
        if self._is_drawing:
            return
        self._is_drawing = True
        self._active_stroke = Stroke(self.color, self.brush_size)
        self._smoother.reset()
        self._snapshot()   # Save state before new stroke (for undo)

    def add_point(self, raw_pt: Tuple[int,int]) -> None:
        """Add a smoothed point to the current stroke."""
        if not self._is_drawing or self._active_stroke is None:
            return

        # Clip to drawing area (don't draw over toolbar)
        if raw_pt[1] < TOOLBAR_HEIGHT:
            return

        pt = self._smoother.smooth(raw_pt)
        self._active_stroke.add_point(pt)

        # Draw interpolated segment on canvas
        if self._last_pt is not None:
            col = (0, 0, 0) if self._eraser_mode else self.color
            sz  = 30        if self._eraser_mode else self.brush_size
            cv2.line(self._canvas, self._last_pt, pt, col, sz, cv2.LINE_AA)
        else:
            col = (0, 0, 0) if self._eraser_mode else self.color
            sz  = 30        if self._eraser_mode else self.brush_size
            cv2.circle(self._canvas, pt, sz // 2, col, -1, cv2.LINE_AA)

        self._last_pt = pt

    def end_stroke(self) -> Optional[Stroke]:
        """
        Finalise the active stroke.
        Returns the completed Stroke if it meets the minimum point threshold,
        else None (too short — likely accidental).
        """
        if not self._is_drawing:
            return None
        self._is_drawing = False
        self._last_pt = None
        stroke = self._active_stroke
        self._active_stroke = None

        if stroke and stroke.is_valid():
            self._completed_strokes.append(stroke)
            return stroke
        return None

    # ── Canvas operations ───────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear the canvas (also resets stroke history)."""
        self._snapshot()
        self._canvas = self._blank_canvas()
        self._completed_strokes.clear()
        self._last_pt = None

    def draw_beautified_shape(self, shape_type: str, points: list,
                               color: Tuple = (0, 255, 140), thickness: int = 3) -> None:
        """Replace the last rough stroke with a clean vector version."""
        if not points:
            return
        pts = np.array(points, dtype=np.int32)
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

        if shape_type == "circle":
            dists = np.sqrt(((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2))
            r = int(np.mean(dists))
            cv2.circle(self._canvas, (cx, cy), r, color, thickness, cv2.LINE_AA)

        elif shape_type in ("square", "rectangle"):
            x, y, w, h = cv2.boundingRect(pts)
            side = max(w, h)
            cv2.rectangle(self._canvas,
                          (cx - side // 2, cy - side // 2),
                          (cx + side // 2, cy + side // 2),
                          color, thickness, cv2.LINE_AA)

        elif shape_type == "triangle":
            hull = cv2.convexHull(pts)
            cv2.drawContours(self._canvas, [hull], 0, color, thickness, cv2.LINE_AA)

        elif shape_type == "line":
            cv2.line(self._canvas, tuple(pts[0]), tuple(pts[-1]), color, thickness, cv2.LINE_AA)

    def overlay_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the drawing canvas on top of the webcam frame.
        Black pixels are treated as transparent.
        """
        out = frame.copy()
        mask = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY) > 10
        out[mask] = cv2.addWeighted(
            out[mask], 0.15,
            self._canvas[mask], 0.85, 0
        )
        return out

    # ── Color / Brush ────────────────────────────────────────────────────────

    def next_color(self) -> Tuple:
        self._color_idx = (self._color_idx + 1) % len(PALETTE)
        return self.color

    def prev_color(self) -> Tuple:
        self._color_idx = (self._color_idx - 1) % len(PALETTE)
        return self.color

    def next_brush(self) -> int:
        self._brush_idx = (self._brush_idx + 1) % len(BRUSH_SIZES)
        return self.brush_size

    def toggle_eraser(self) -> bool:
        self._eraser_mode = not self._eraser_mode
        return self._eraser_mode

    # ── Undo / Redo ──────────────────────────────────────────────────────────

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._canvas.copy())
        self._canvas = self._undo_stack.pop()
        # Resyncing stroke list is expensive; clear for safety
        self._completed_strokes.clear()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._canvas.copy())
        self._canvas = self._redo_stack.pop()
        return True

    def save(self, path: str) -> bool:
        """Save the current canvas as a PNG."""
        try:
            cv2.imwrite(path, self._canvas)
            return True
        except Exception:
            return False

    # ── Internals ────────────────────────────────────────────────────────────

    def _blank_canvas(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = CANVAS_BG_COLOR
        return img

    def _snapshot(self) -> None:
        """Push current canvas state to undo stack."""
        self._undo_stack.append(self._canvas.copy())
        self._redo_stack.clear()   # Any new action discards redo history
