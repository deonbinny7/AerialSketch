"""
hand_tracking/tracker.py
========================
Encapsulates MediaPipe Hands processing.

Responsibilities
----------------
* Detect hand landmarks from a BGR frame.
* Classify the current hand gesture (DRAWING, PAUSED, CLEAR, IDLE).
* Expose the smoothed fingertip position used for drawing.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import (
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    DRAWING_FINGER_ID, MIDDLE_FINGER_ID, THUMB_TIP_ID, INDEX_MCP_ID,
    PINCH_THRESHOLD_PX, GESTURE_DEBOUNCE_FRAMES, CLEAR_HOLD_FRAMES,
    SHOW_LANDMARKS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Gesture States
# ─────────────────────────────────────────────────────────────────────────────

class Gesture(Enum):
    IDLE     = auto()   # No hand / unknown pose
    DRAWING  = auto()   # Index finger up, others curled
    PAUSED   = auto()   # Index + middle up (V-sign / peace)
    CLEAR    = auto()   # Open palm held for N frames


# ─────────────────────────────────────────────────────────────────────────────
#  Data container returned each frame
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackingResult:
    detected:   bool              = False
    gesture:    Gesture           = Gesture.IDLE
    fingertip:  Optional[Tuple[int,int]] = None   # (x, y) pixel coords
    landmarks:  Optional[object]  = None           # raw NormalizedLandmarkList
    hand_bb:    Optional[Tuple]   = None           # (x,y,w,h) bounding box


# ─────────────────────────────────────────────────────────────────────────────
#  Finger-up helper
# ─────────────────────────────────────────────────────────────────────────────

_FINGER_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky
_FINGER_PIPS = [6, 10, 14, 18]   # corresponding PIP joints


def _count_fingers_up(lm, img_h: int, img_w: int) -> list[bool]:
    """Return a list of booleans [index, middle, ring, pinky] for each
    finger that is currently extended (tip above pip)."""
    up = []
    for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS):
        tip_y = lm.landmark[tip].y * img_h
        pip_y = lm.landmark[pip].y * img_h
        up.append(tip_y < pip_y)   # y=0 is top of frame
    return up


def _thumb_up(lm, img_w: int) -> bool:
    """Rough thumb-extended check (horizontal movement)."""
    tip_x  = lm.landmark[THUMB_TIP_ID].x  * img_w
    mcp_x  = lm.landmark[INDEX_MCP_ID].x  * img_w
    return abs(tip_x - mcp_x) > 30


# ─────────────────────────────────────────────────────────────────────────────
#  HandTracker
# ─────────────────────────────────────────────────────────────────────────────

class HandTracker:
    """Wraps MediaPipe Hands and classifies gestures each frame."""

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        # Debounce buffers
        self._gesture_buf: deque[Gesture] = deque(maxlen=GESTURE_DEBOUNCE_FRAMES)
        self._clear_counter: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> TrackingResult:
        """Process one BGR frame and return a TrackingResult."""
        h, w = bgr_frame.shape[:2]

        # MediaPipe requires RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not mp_result.multi_hand_landmarks:
            self._gesture_buf.clear()
            self._clear_counter = 0
            return TrackingResult(detected=False)

        lm = mp_result.multi_hand_landmarks[0]

        # Fingertip pixel coords
        tip = lm.landmark[DRAWING_FINGER_ID]
        fx = int(tip.x * w)
        fy = int(tip.y * h)

        # Gesture classification
        raw_gesture = self._classify(lm, h, w)
        stable_gesture = self._debounce(raw_gesture)

        # Bounding box
        xs = [int(l.x * w) for l in lm.landmark]
        ys = [int(l.y * h) for l in lm.landmark]
        bb = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        if SHOW_LANDMARKS:
            self._mp_draw.draw_landmarks(
                bgr_frame, lm, self._mp_hands.HAND_CONNECTIONS
            )

        return TrackingResult(
            detected=True,
            gesture=stable_gesture,
            fingertip=(fx, fy),
            landmarks=lm,
            hand_bb=bb,
        )

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    # ── Private helpers ─────────────────────────────────────────────────────

    def _classify(self, lm, img_h: int, img_w: int) -> Gesture:
        """Map landmark configuration → Gesture enum."""
        fingers = _count_fingers_up(lm, img_h, img_w)
        idx, mid, rng, pnk = fingers

        # Open palm = all four fingers + thumb roughly extended
        if idx and mid and rng and pnk and _thumb_up(lm, img_w):
            return Gesture.CLEAR

        # V-sign (two fingers) = pause
        if idx and mid and not rng and not pnk:
            return Gesture.PAUSED

        # Index only = drawing
        if idx and not mid and not rng and not pnk:
            return Gesture.DRAWING

        return Gesture.IDLE

    def _debounce(self, raw: Gesture) -> Gesture:
        """Require N consecutive frames of the same gesture before switching."""
        self._gesture_buf.append(raw)
        if len(self._gesture_buf) < GESTURE_DEBOUNCE_FRAMES:
            return Gesture.IDLE

        # All frames agree?
        if all(g == raw for g in self._gesture_buf):
            return raw
        # Return the most common gesture in the buffer as a fallback
        counts = {}
        for g in self._gesture_buf:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get)
