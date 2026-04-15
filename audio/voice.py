"""
audio/voice.py
==============
Non-blocking, queue-based TTS engine.
Speaks shape announcements on a background daemon thread so the
main loop (60 FPS) is never stalled by speech synthesis.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

from config import (
    AUDIO_ENABLED, TTS_RATE, TTS_VOLUME, AUDIO_QUEUE_MAX,
)


class VoiceFeedback:
    """
    Thread-safe TTS wrapper.

    Usage
    -----
    vf = VoiceFeedback()
    vf.start()
    vf.say("Circle detected")
    ...
    vf.stop()
    """

    def __init__(self) -> None:
        self._enabled  = AUDIO_ENABLED
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
        self._thread: Optional[threading.Thread] = None
        self._running  = False
        self._engine   = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise pyttsx3 and start the background worker thread."""
        if not self._enabled:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Gracefully shut down the TTS thread."""
        if not self._running:
            return
        self._running = False
        # Send sentinel to unblock the worker
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=3)

    # ── Public API ───────────────────────────────────────────────────────────

    def say(self, text: str) -> None:
        """
        Queue a TTS utterance. If the queue is full (burst of detections),
        drop the oldest item to keep latency low.
        """
        if not self._enabled or not self._running:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            # Discard oldest, insert new
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                pass

    def announce_shape(self, shape: str, confidence: float) -> None:
        """Convenience wrapper that formats the shape announcement."""
        pct = int(confidence * 100)
        self.say(f"{shape.capitalize()} detected — {pct} percent confidence")

    # ── Worker (runs on background thread) ──────────────────────────────────

    def _worker(self) -> None:
        """Background thread that drains the queue and speaks each item."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
            self._engine = engine

            while self._running:
                text = self._queue.get()           # blocks until available
                if text is None:                   # sentinel → shut down
                    break
                engine.say(text)
                engine.runAndWait()

        except ImportError:
            print("[voice] pyttsx3 not installed — voice feedback disabled.")
        except Exception as exc:
            print(f"[voice] TTS error: {exc}")
