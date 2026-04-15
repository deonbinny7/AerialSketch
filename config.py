# =============================================================================
#  AerialSketch — Centralized Configuration  v2.0 (Production Upgrade)
#  All constants live here. Import from this file ONLY. No hardcoding elsewhere.
# =============================================================================

# ── Camera / Input ────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0          # 0 = default webcam
CAMERA_WIDTH: int = 1280       # Capture width  (pixels)
CAMERA_HEIGHT: int = 720       # Capture height (pixels)
CAMERA_FPS: int = 60           # Requested capture FPS
FLIP_FRAME: bool = True        # Mirror the frame (natural hand-UX)

# ── Hand Tracking (MediaPipe) ──────────────────────────────────────────────────
MAX_HANDS: int = 1                    # Detect at most 1 hand
DETECTION_CONFIDENCE: float = 0.75   # Minimum detection confidence
TRACKING_CONFIDENCE: float = 0.75    # Minimum tracking confidence
DRAWING_FINGER_ID: int = 8           # Index fingertip landmark
MIDDLE_FINGER_ID: int = 12           # Middle fingertip landmark
THUMB_TIP_ID: int = 4                # Thumb tip landmark
INDEX_MCP_ID: int = 5                # Index finger MCP (knuckle)
PINCH_THRESHOLD_PX: int = 45        # px distance to trigger pinch (pause/resume)
GESTURE_DEBOUNCE_FRAMES: int = 6    # Frames a gesture must persist before firing
CLEAR_HOLD_FRAMES: int = 25         # Frames open-palm must hold to clear canvas

# ── Drawing ────────────────────────────────────────────────────────────────────
SMOOTHING_WINDOW: int = 7           # Moving-average window (1 = no smoothing)
MIN_STROKE_POINTS: int = 15         # Ignore strokes shorter than this
MAX_STROKE_POINTS: int = 600        # Decimate strokes longer than this
DEFAULT_COLOR_IDX: int = 0          # Default color index into PALETTE

PALETTE: list = [
    (0,   200, 255),   # 0 — Amber
    (0,   255, 140),   # 1 — Mint
    (240,  80, 255),   # 2 — Violet
    (80,  180, 255),   # 3 — Sky
]
PALETTE_NAMES: list = ["Amber", "Mint", "Violet", "Sky"]

BRUSH_SIZES: list = [3, 5, 8, 12]   # px — cycle with gesture
DEFAULT_BRUSH_IDX: int = 1

ERASER_SIZE: int = 30               # px — eraser radius
CANVAS_BG_COLOR: tuple = (0, 0, 0)  # Pure black — must be below overlay mask threshold (> 10)

# ── Drawing Quality (Upgrade #4) ───────────────────────────────────────────────
STROKE_GAP_FILL_PX: int = 12        # Max gap (px) the point interpolator will bridge.
                                     # Prevents hollow circles from fast hand movement.

# ── Recognition ────────────────────────────────────────────────────────────────
RECOG_IMAGE_SIZE: int = 64           # Normalize strokes to this square (px)
RECOG_CONFIDENCE_THRESHOLD: float = 0.55  # Min confidence to announce shape (raised from 0.52 to reduce false positives)
RECOG_BACKEND: str = "rule_based"   # "rule_based" | "ml" (set to 'ml' if TF is correctly set up)
SHAPE_CLASSES: list = ["circle", "square", "triangle", "line"]
SHAPE_BEAUTIFY: bool = True          # Replace sketch with clean vector on detection
AUTO_CLEAR_AFTER_DETECT: bool = False  # Clear canvas after shape announced
RECOGNIZED_SHAPE_COLOR: tuple = (0, 255, 140)   # BGR color for beautified shape overlay
BEAUTIFY_THICKNESS: int = 3                      # Stroke thickness of the beautified shape

# ── Hybrid Fusion & Scoring (Upgrade #5 & #6) ──────────────────────────────────
FUSION_WEIGHT_RULES: float = 0.6    # Rule-based contribution to hybrid score
FUSION_WEIGHT_ML: float = 0.4       # ML contribution to hybrid score

# Confidence Engine constants
HYBRID_CONFIDENCE_FLOOR: float = 0.35  # After fusion, scores below this are treated as uncertain
AMBIGUITY_MARGIN: float = 0.15         # If top-2 scores are within this margin, apply penalty
PARTIAL_SHAPE_PENALTY: float = 0.40    # Multiplicative penalty for open/broken/partial shapes

# ── Detection Timing (Upgrade #7) ─────────────────────────────────────────────
# Tuned up from 0.6s → 0.7s: gives more deliberate strokes time to complete
# before triggering classification, reducing premature classification.
DETECTION_TRIGGER_DELAY: float = 0.7   # Seconds of inactivity before triggering detection

# ── Preprocessing (Upgrade #3) ────────────────────────────────────────────────
CONTOUR_MIN_AREA: int = 1200           # Noise floor — contours smaller than this are ignored (was 800)
CLOSE_GAP_RATIO: float = 0.25         # Auto-close stroke if gap < this fraction of bbox diagonal

# Ideal metric target values for the improved heuristic engine (Upgrade #1)
# NOTE: circle circularity target reduced from 0.88 → 0.82 because hand-drawn
#       circles rarely achieve near-perfect roundness. This avoids penalizing
#       good circles too harshly.
IDEAL_METRICS: dict = {
    "circle":   {"circularity": 0.82, "solidity": 0.93, "vertices": 8},
    "square":   {"circularity": 0.73, "solidity": 0.92, "vertices": 4},
    "triangle": {"circularity": 0.58, "solidity": 0.88, "vertices": 3},
    "line":     {"straightness": 0.90, "vertices": 2},
}

# ── Audio ──────────────────────────────────────────────────────────────────────
AUDIO_ENABLED: bool = True
TTS_RATE: int = 165         # Words-per-minute
TTS_VOLUME: float = 0.9    # 0.0–1.0
AUDIO_QUEUE_MAX: int = 3   # Max queued TTS events

# ── UI / Display ───────────────────────────────────────────────────────────────
WINDOW_TITLE: str = "AerialSketch ✦"
SHOW_FPS: bool = True
SHOW_LANDMARKS: bool = False        # Debug: render MediaPipe skeleton
SHOW_GESTURE_LABEL: bool = True
GLOW_RADIUS: int = 18              # Fingertip glow radius (px)
TOOLBAR_HEIGHT: int = 80           # Top toolbar height (px)
OVERLAY_ALPHA: float = 0.30        # Toolbar background alpha

# ── Advanced Features ──────────────────────────────────────────────────────────
UNDO_HISTORY_LIMIT: int = 20       # Max undo steps stored
SAVE_OUTPUT_DIR: str = "saved/"    # Directory for saved drawings

# ── Logging / Debug ────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR
DEBUG_MODE: bool = False  # Toggle at runtime with 'd' key
