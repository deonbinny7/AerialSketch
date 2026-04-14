"""
recognition/predictor.py
=========================
Unified prediction interface. Selects the backend based on config
and exposes a single predict() method.

Backends
--------
* rule_based  – Fast geometric heuristics. Zero dependencies beyond NumPy/OpenCV.
* ml          – CNN inference via TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import (
    RECOG_BACKEND, RECOG_CONFIDENCE_THRESHOLD,
    SHAPE_CLASSES, RECOG_IMAGE_SIZE,
)
from recognition.preprocess import stroke_to_image, extract_geometric_features


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    shape:      str               # e.g. "circle"
    confidence: float             # 0.0–1.0
    accepted:   bool              # True if confidence ≥ threshold
    all_scores: dict              # {shape_name: score}


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-Based Recogniser
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_predict(points: List[Tuple[int,int]]) -> PredictionResult:
    """
    Classify a stroke using hand-crafted geometric rules.

    Rules
    -----
    LINE      straightness > 0.75
    CIRCLE    circularity  > 0.50
    TRIANGLE  corners ≈ 3  AND aspect close to 1.0
    SQUARE    corners ≈ 4  AND aspect close to 1.0
    """
    feats = extract_geometric_features(points)
    if not feats:
        return PredictionResult("unknown", 0.0, False, {})

    s   = feats["straightness"]
    c   = feats["circularity"]
    ar  = feats["aspect_ratio"]
    nc  = feats["num_corners"]

    scores = {"circle": 0.0, "square": 0.0, "triangle": 0.0, "line": 0.0}

    # LINE ─────────────────────────────────────────────────────────────────
    scores["line"] = min(1.0, s * 1.3)

    # CIRCLE ───────────────────────────────────────────────────────────────
    # High circularity + ≥5 hull corners (smooth curve)
    circ_score = min(1.0, c * 2.0)
    corner_penalty = max(0, (nc - 12) * 0.05)   # penalise if too angular
    scores["circle"] = max(0.0, circ_score - corner_penalty)

    # TRIANGLE ─────────────────────────────────────────────────────────────
    corner_match_3 = max(0.0, 1.0 - abs(nc - 3) * 0.25)
    scores["triangle"] = corner_match_3 * min(1.0, 1.0 / (abs(ar - 1.0) + 0.5))

    # SQUARE ───────────────────────────────────────────────────────────────
    corner_match_4 = max(0.0, 1.0 - abs(nc - 4) * 0.25)
    aspect_match   = max(0.0, 1.0 - abs(ar - 1.0))
    scores["square"] = corner_match_4 * aspect_match

    # If straightness is very high, suppress shape scores
    if s > 0.80:
        for k in ("circle", "square", "triangle"):
            scores[k] *= (1.0 - s)

    best_shape = max(scores, key=scores.__getitem__)
    confidence = scores[best_shape]
    accepted   = confidence >= RECOG_CONFIDENCE_THRESHOLD

    return PredictionResult(best_shape, round(confidence, 3), accepted, scores)


# ─────────────────────────────────────────────────────────────────────────────
#  ML (CNN) Backend
# ─────────────────────────────────────────────────────────────────────────────

_cnn_model = None   # Lazy-loaded singleton


def _load_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        from recognition.model import load_model, train_and_save
        _cnn_model = load_model()
        if _cnn_model is None:
            print("[predictor] No saved CNN found — training from scratch …")
            train_and_save()
            _cnn_model = load_model()
    return _cnn_model


def _ml_predict(points: List[Tuple[int,int]]) -> PredictionResult:
    model = _load_cnn_model()
    if model is None:
        # Fallback if TF unavailable
        return _rule_based_predict(points)

    img = stroke_to_image(points, RECOG_IMAGE_SIZE)
    x   = img.astype(np.float32) / 255.0
    x   = np.expand_dims(x, 0)   # (1, size, size, 1)

    probs = model.predict(x, verbose=0)[0]
    label_idx  = int(np.argmax(probs))
    confidence = float(probs[label_idx])
    shape      = SHAPE_CLASSES[label_idx]
    all_scores = {SHAPE_CLASSES[i]: float(probs[i]) for i in range(len(SHAPE_CLASSES))}

    accepted = confidence >= RECOG_CONFIDENCE_THRESHOLD
    return PredictionResult(shape, round(confidence, 3), accepted, all_scores)


# ─────────────────────────────────────────────────────────────────────────────
#  Unified Predictor
# ─────────────────────────────────────────────────────────────────────────────

class ShapePredictor:
    """
    Single entry-point for shape classification.
    Selects backend from config.RECOG_BACKEND.
    """

    def __init__(self) -> None:
        self._backend = RECOG_BACKEND
        print(f"[predictor] Using '{self._backend}' recognition backend.")

    def predict(self, points: List[Tuple[int,int]]) -> Optional[PredictionResult]:
        """
        Classify the drawn points.

        Parameters
        ----------
        points : list of (x, y) pixel tuples — the raw or decimated stroke

        Returns
        -------
        PredictionResult or None (if input is too short)
        """
        from config import MIN_STROKE_POINTS
        if len(points) < MIN_STROKE_POINTS:
            return None

        if self._backend == "ml":
            result = _ml_predict(points)
        else:
            result = _rule_based_predict(points)

        return result
