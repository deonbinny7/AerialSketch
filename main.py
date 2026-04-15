"""
main.py  —  AerialSketch Entry Point
=====================================
Ties every module together in the main application loop.

Architecture
------------
1. Capture frame from webcam
2. Hand tracker → gesture + fingertip
3. Drawing canvas → accumulate / finalize stroke
4. Recognition → classify finished stroke
5. Audio → announce result on background thread
6. UI renderer → composites everything and displays

Keyboard shortcuts
------------------
C  — next color
B  — cycle brush size
Z  — undo
Y  — redo
S  — save current canvas as PNG
E  — toggle eraser
D  — toggle debug mode
Q  — quit
"""

import os
import sys
import datetime
import time

import cv2

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    FLIP_FRAME, WINDOW_TITLE, DEBUG_MODE, SAVE_OUTPUT_DIR,
    SHAPE_BEAUTIFY, AUTO_CLEAR_AFTER_DETECT, RECOGNIZED_SHAPE_COLOR,
    BEAUTIFY_THICKNESS, PALETTE, DETECTION_TRIGGER_DELAY,
)
from hand_tracking.tracker import HandTracker, Gesture
from drawing.canvas        import DrawingCanvas
from recognition.predictor import ShapePredictor
from audio.voice           import VoiceFeedback
from ui.interface          import UIRenderer
from utils.helpers         import FPSCounter, save_drawing

# Try to import LOG_LEVEL, fall back gracefully
try:
    from config import LOG_LEVEL
except ImportError:
    LOG_LEVEL = "INFO"

# ─────────────────────────────────────────────────────────────────────────────
#  Application
# ─────────────────────────────────────────────────────────────────────────────

class AerialSketchApp:
    """Main application controller."""

    def __init__(self) -> None:
        print("[main] Initialising AerialSketch …")

        # Core services
        self.tracker    = HandTracker()
        self.canvas     = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.predictor  = ShapePredictor()
        self.voice      = VoiceFeedback()
        self.ui         = UIRenderer()
        self.fps_ctr    = FPSCounter()

        # Runtime state
        self._prev_gesture  = Gesture.IDLE
        self._debug         = DEBUG_MODE
        self._running       = True

        # Draw-stop detection (Fix 1 + Fix 10)
        self._pending_stroke = None   # stroke waiting to be classified
        self._pending_since  = 0.0   # timestamp when drawing stopped
        self._draw_start     = 0.0   # timestamp when drawing started
        self._STOP_DELAY     = DETECTION_TRIGGER_DELAY
        self._MIN_DRAW_TIME  = 0.4   # ignore strokes shorter than this (seconds)

        # Camera
        self._cap = self._open_camera()

    # ── Camera setup ─────────────────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        if not cap.isOpened():
            sys.exit("[main] Cannot open camera. Check CAMERA_INDEX in config.py")
        return cap

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.voice.start()
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, CAMERA_WIDTH, CAMERA_HEIGHT)
        print("[main] Running! Press Q or ESC to quit.")

        try:
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    print("[main] Frame read failed. Retrying …")
                    continue

                if FLIP_FRAME:
                    frame = cv2.flip(frame, 1)

                # ── 1. Hand tracking ───────────────────────────────────────
                result = self.tracker.process(frame)
                gesture = result.gesture if result.detected else Gesture.IDLE

                # ── 2. Gesture → Drawing state machine ────────────────────
                self._update_drawing(gesture, result.fingertip)

                # ── 3. Composite canvas over frame ────────────────────────
                frame = self.canvas.overlay_on_frame(frame)

                # ── 4. UI rendering ───────────────────────────────────────
                fps = self.fps_ctr.tick()
                frame = self.ui.render(
                    frame=frame,
                    fps=fps,
                    gesture_label=gesture.name,
                    color_idx=self.canvas._color_idx,
                    brush_idx=self.canvas._brush_idx,
                    fingertip=result.fingertip,
                    is_drawing=(gesture == Gesture.DRAWING),
                    debug=self._debug,
                )

                cv2.imshow(WINDOW_TITLE, frame)

                # ── 5. Keyboard input ─────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)

        finally:
            self._shutdown()

    # ── Drawing state machine ─────────────────────────────────────────────────

    def _update_drawing(self, gesture: Gesture, fingertip) -> None:
        prev = self._prev_gesture
        now  = time.time()

        # ── Enter DRAWING ──────────────────────────────────────────────────
        if gesture == Gesture.DRAWING:
            if prev != Gesture.DRAWING:
                self.canvas.begin_stroke()
                self._draw_start     = now        # record start time
                self._pending_stroke = None       # cancel any pending old stroke
            if fingertip:
                self.canvas.add_point(fingertip)

        # ── Leave DRAWING → queue stroke for delayed classification ────────
        elif prev == Gesture.DRAWING:
            draw_duration = now - self._draw_start
            stroke = self.canvas.end_stroke()
            if stroke and stroke.is_valid() and draw_duration >= self._MIN_DRAW_TIME:
                self._pending_stroke = stroke
                self._pending_since  = now        # start the stop-delay timer
            # else: too short / accidental — discard silently

        # ── CLEAR canvas ───────────────────────────────────────────────────
        elif gesture == Gesture.CLEAR and prev != Gesture.CLEAR:
            self._pending_stroke = None
            self.canvas.clear()
            self.ui.show_toast("Canvas cleared", color=(255, 80, 80))

        # ── CLASSIFY pending stroke after stop delay ────────────────────────
        if (self._pending_stroke is not None
                and gesture != Gesture.DRAWING
                and (now - self._pending_since) >= self._STOP_DELAY):
            stroke = self._pending_stroke
            self._pending_stroke = None
            pts    = stroke.decimated()
            result = self.predictor.predict(pts)
            if result and result.accepted:
                self._on_shape_detected(stroke, result)

        self._prev_gesture = gesture


    def _on_shape_detected(self, stroke, result) -> None:
        shape      = result.shape
        confidence = result.confidence

        # Beautify stroke — first wipe the raw freehand lines, then draw clean vector
        if SHAPE_BEAUTIFY:
            self.canvas.clear_strokes_only()   # remove raw residue, keep canvas bg
            self.canvas.draw_beautified_shape(
                shape,
                stroke.points,
                color=RECOGNIZED_SHAPE_COLOR,
                thickness=BEAUTIFY_THICKNESS,
            )

        # UI badge
        self.ui.set_detection(shape, confidence)
        self.ui.show_toast(
            f"{shape.capitalize()} — {int(confidence*100)}%",
            color=(0, 200, 255),
        )

        # Voice
        self.voice.announce_shape(shape, confidence)

        # Auto-clear
        if AUTO_CLEAR_AFTER_DETECT:
            self.canvas.clear()

        print(f"[detect] {shape.upper():10s}  conf={confidence:.2f}  "
              f"pts={len(stroke.points)}")

    # ── Keyboard handler ──────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> None:
        if key in (ord('q'), ord('Q'), 27):          # ESC or Q → quit
            self._running = False

        elif key in (ord('c'), ord('C')):             # C → next color
            col = self.canvas.next_color()
            self.ui.show_toast(f"Color: {PALETTE[self.canvas._color_idx]}",
                               color=col)

        elif key in (ord('b'), ord('B')):             # B → brush size
            sz = self.canvas.next_brush()
            self.ui.show_toast(f"Brush: {sz}px")

        elif key in (ord('z'), ord('Z')):             # Z → undo
            ok = self.canvas.undo()
            self.ui.show_toast("Undo" if ok else "Nothing to undo",
                               color=(255, 200, 0) if ok else (255, 80, 80))

        elif key in (ord('y'), ord('Y')):             # Y → redo
            ok = self.canvas.redo()
            self.ui.show_toast("Redo" if ok else "Nothing to redo",
                               color=(255, 200, 0) if ok else (255, 80, 80))

        elif key in (ord('s'), ord('S')):             # S → save
            path = save_drawing(self.canvas.canvas, SAVE_OUTPUT_DIR)
            self.ui.show_toast(f"Saved: {os.path.basename(path)}",
                               color=(0, 255, 140))
            print(f"[main] Drawing saved → {path}")

        elif key in (ord('e'), ord('E')):             # E → eraser
            erasing = self.canvas.toggle_eraser()
            self.ui.show_toast(
                "Eraser ON" if erasing else "Brush mode",
                color=(255, 160, 0) if erasing else (0, 200, 255),
            )

        elif key in (ord('d'), ord('D')):             # D → debug
            self._debug = not self._debug
            self.ui.show_toast(
                f"Debug {'ON' if self._debug else 'OFF'}",
                color=(255, 80, 80),
            )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        print("[main] Shutting down …")
        self.voice.stop()
        self.tracker.release()
        self._cap.release()
        cv2.destroyAllWindows()
        print("[main] Goodbye!")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(SAVE_OUTPUT_DIR, exist_ok=True)
    app = AerialSketchApp()
    app.run()
