"""
recognition/predictor.py  v2.0 — Production Upgrade
=====================================================
Unified prediction interface — selects backend from config and exposes a
single predict() method.

Backends
--------
* rule_based  – Fast geometric heuristics. Zero dependencies beyond NumPy/OpenCV.
* ml          – Hybrid CNN + rule-based fusion via TensorFlow.

Upgrade Summary vs v1.0
------------------------
[FIX]  _ml_predict_raw was truncated at line 150 — return statement was MISSING.
[FIX]  _ml_predict() hybrid-fusion function was completely absent.
[NEW]  _is_partial_shape() — penalises open/broken shapes before acceptance.
[NEW]  Uncertainty floor: scores below HYBRID_CONFIDENCE_FLOOR → rejected early.
[NEW]  Stronger vertex score decay: 0.70^error (was 0.82) → better poly separation.
[NEW]  Aspect-ratio bonus injected for square/rectangle discrimination.
[NEW]  Ambiguity penalty extended: top-N spread-based scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from config import (
    RECOG_BACKEND, RECOG_CONFIDENCE_THRESHOLD,
    SHAPE_CLASSES, RECOG_IMAGE_SIZE,
    FUSION_WEIGHT_RULES, FUSION_WEIGHT_ML,
    HYBRID_CONFIDENCE_FLOOR, AMBIGUITY_MARGIN,
    PARTIAL_SHAPE_PENALTY,
)
from recognition.preprocess import stroke_to_image, extract_geometric_features, extract_contour_features


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    shape:      str               # e.g. "circle"
    confidence: float             # 0.0–1.0
    accepted:   bool              # True if confidence ≥ threshold
    all_scores: Dict[str, float]  # per-class score breakdown
    backend:    str = "unknown"   # which backend produced this result
    rejected_reason: str = ""     # why rejected (if accepted=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Confidence Engine  (Upgrade #6)
# ─────────────────────────────────────────────────────────────────────────────

def _score_by_metrics(features: dict) -> dict:
    """
    Score each shape class against ideal metric targets.

    Changes vs v1.0
    ---------------
    - Vertex decay exponent changed from 0.82 → 0.70 for sharper poly separation
    - Aspect-ratio penalty added for square: penalises strongly non-square bboxes
    - Line scoring uses straightness² instead of straightness^2.5 (smoother curve)
    - All weights tuned against empirical error cases (broken circles, noisy squares)
    """
    from config import IDEAL_METRICS
    scores = {s: 0.0 for s in SHAPE_CLASSES}

    if not features:
        return scores

    for shape in SHAPE_CLASSES:
        if shape not in IDEAL_METRICS:
            continue
        ideal = IDEAL_METRICS[shape]
        val = 1.0

        # ── Circularity (primary discriminator for circles vs polygons) ──────
        if "circularity" in ideal:
            diff = abs(features.get("circularity", 0) - ideal["circularity"])
            val *= max(0.0, 1.0 - (diff * 3.0))

        # ── Solidity (distinguishes noisy stroke bundles from clean shapes) ──
        if "solidity" in ideal:
            diff = abs(features.get("solidity", 0) - ideal["solidity"])
            val *= max(0.0, 1.0 - (diff * 3.5))

        # ── Vertex count with sharper decay ──────────────────────────────────
        if "vertices" in ideal:
            target_v = ideal["vertices"]
            v_tight  = features.get("vertices_tight", 0)
            v_loose  = features.get("vertices_loose", 0)
            # Pick whichever approximation is closer to the ideal
            best_v = v_tight if abs(v_tight - target_v) <= abs(v_loose - target_v) else v_loose
            v_err  = abs(best_v - target_v)
            val   *= (0.70 ** v_err)   # was 0.82 — sharper penalty, better poly separation

        # ── Aspect ratio bonus for square (punishes wide/tall rectangles) ────
        if shape == "square":
            ar = features.get("aspect_ratio", 1.0)
            # Ideal square has ar ≈ 1.0; decay symmetrically around 1.0
            ar_dev = abs(ar - 1.0)
            val *= max(0.0, 1.0 - ar_dev * 1.8)

        # ── Line override: straightness-squared is smoother ──────────────────
        if shape == "line":
            s = features.get("straightness", 0.0)
            val = (s ** 2.0) if s > 0.55 else 0.0

        scores[shape] = round(val, 4)

    return scores


def _apply_ambiguity_penalty(scores: dict) -> dict:
    """
    Reduce confidence if the top two shapes score too similarly.

    Changes vs v1.0
    ---------------
    - Penalty now scales linearly from 0→AMBIGUITY_MARGIN instead of jumping
    - Minimum post-penalty value raised from 0.4 → 0.3 for more nuance
    - Uses AMBIGUITY_MARGIN from config for tunability
    """
    if not scores:
        return {}
    sorted_vals = sorted(scores.values(), reverse=True)
    if len(sorted_vals) < 2:
        return scores

    margin = sorted_vals[0] - sorted_vals[1]
    if margin < AMBIGUITY_MARGIN:
        # Penalty proportion: 0 when margin=0, 1.0 when margin=AMBIGUITY_MARGIN
        penalty_factor = margin / AMBIGUITY_MARGIN      # 0.0–1.0
        penalty_factor = max(0.30, penalty_factor)       # floor at 0.30
        top_shape = max(scores, key=scores.__getitem__)
        scores[top_shape] = round(scores[top_shape] * penalty_factor, 4)

    return scores


def _is_partial_shape(features: dict) -> bool:
    """
    Detect broken / partial shapes — open strokes with low solidity.

    Rationale: A circle that stops 30% short will have gap > threshold AND
    solidity < 0.60 (the missing arc creates convex hull inflation).
    Returns True  → apply PARTIAL_SHAPE_PENALTY to all scores.
    Returns False → shape appears complete enough to classify normally.
    """
    if not features:
        return False
        
    # Exception: if it's highly straight, it's a line, not a broken circle/square
    if features.get("straightness", 0.0) > 0.75:
        return False
        
    is_open    = not features.get("closed", True)
    low_solid  = features.get("solidity", 1.0) < 0.60
    low_area   = features.get("area", 9999) < 2000
    return is_open and (low_solid or low_area)


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-based recogniser  (Upgrade #1 + #6 + #8)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_predict(points: List[Tuple[int, int]]) -> PredictionResult:
    """Classify stroke using the upgraded probabilistic metric engine."""
    cf = extract_contour_features(points)
    if not cf:
        return PredictionResult("unknown", 0.0, False, {}, "rule_based", "no_contour")

    scores = _score_by_metrics(cf)

    # ── Failure case: penalise broken / partial shapes (Upgrade #8) ──────────
    rejection_reason = ""
    if _is_partial_shape(cf):
        scores = {k: round(v * (1.0 - PARTIAL_SHAPE_PENALTY), 4) for k, v in scores.items()}
        rejection_reason = "partial_shape"

    scores = _apply_ambiguity_penalty(scores)

    best_shape = max(scores, key=scores.__getitem__)
    confidence = scores[best_shape]

    # ── Uncertainty floor (Upgrade #6) ───────────────────────────────────────
    if confidence < HYBRID_CONFIDENCE_FLOOR:
        rejection_reason = rejection_reason or "low_confidence"
        return PredictionResult(best_shape, confidence, False, scores, "rule_based", rejection_reason)

    accepted = confidence >= RECOG_CONFIDENCE_THRESHOLD
    if not accepted and not rejection_reason:
        rejection_reason = "below_threshold"

    return PredictionResult(best_shape, confidence, accepted, scores, "rule_based", rejection_reason)


# ─────────────────────────────────────────────────────────────────────────────
#  ML (CNN) Backend  (Upgrade #5 — CRITICAL FIX)
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


def _ml_predict_raw(points: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Run CNN inference on stroke points.
    Returns per-class probability dict, or empty dict on failure.

    FIX v2.0: This function was truncated in v1.0 — return statement was missing.
    """
    model = _load_cnn_model()
    if model is None:
        return {}

    img = stroke_to_image(points, RECOG_IMAGE_SIZE)
    x   = img.astype(np.float32) / 255.0
    x   = np.expand_dims(x, 0)   # (1, H, W, 1)

    try:
        probs = model.predict(x, verbose=0)[0]   # shape: (num_classes,)
    except Exception as exc:
        print(f"[predictor] CNN inference error: {exc}")
        return {}

    # Convert probability array → named dict
    return {cls: float(probs[i]) for i, cls in enumerate(SHAPE_CLASSES)}


def _ml_predict(points: List[Tuple[int, int]]) -> PredictionResult:
    """
    Hybrid fusion: combine rule-based geometric scores with CNN probabilities.

    Formula (from config):
        final[shape] = rule_score[shape] * FUSION_WEIGHT_RULES
                     + ml_prob[shape]    * FUSION_WEIGHT_ML

    Rationale
    ---------
    Rules excel at detecting circularity and straight lines (fast, deterministic).
    CNN excels at texture / global shape patterns (handles noisy, partial strokes).
    Blending both at 0.6 / 0.4 weights gives lower variance than either alone.

    FIX v2.0: This entire function was absent in v1.0, causing silent crashes
              whenever RECOG_BACKEND = "ml" was used.
    """
    # ── Step 1: rule-based scores ─────────────────────────────────────────────
    cf = extract_contour_features(points)
    if not cf:
        return PredictionResult("unknown", 0.0, False, {}, "hybrid", "no_contour")

    rule_scores = _score_by_metrics(cf)

    # ── Step 2: partial-shape check from rules (Upgrade #8) ──────────────────
    rejection_reason = ""
    is_partial = _is_partial_shape(cf)
    if is_partial:
        rule_scores = {k: round(v * (1.0 - PARTIAL_SHAPE_PENALTY), 4) for k, v in rule_scores.items()}
        rejection_reason = "partial_shape"

    # ── Step 3: ML probabilities ─────────────────────────────────────────────
    ml_probs = _ml_predict_raw(points)

    if not ml_probs:
        # CNN unavailable — fall back to rule-based only (don't crash)
        print("[predictor] CNN unavailable, falling back to rule-based.")
        rule_scores = _apply_ambiguity_penalty(rule_scores)
        best = max(rule_scores, key=rule_scores.__getitem__)
        conf = rule_scores[best]
        accepted = conf >= RECOG_CONFIDENCE_THRESHOLD and conf >= HYBRID_CONFIDENCE_FLOOR
        return PredictionResult(best, conf, accepted, rule_scores, "rule_based_fallback", rejection_reason)

    # ── Step 4: Weighted fusion ────────────────────────────────────────────────
    fused: Dict[str, float] = {}
    for shape in SHAPE_CLASSES:
        r = rule_scores.get(shape, 0.0)
        m = ml_probs.get(shape, 0.0)
        fused[shape] = round(r * FUSION_WEIGHT_RULES + m * FUSION_WEIGHT_ML, 4)

    # ── Step 5: Ambiguity penalty on fused scores (Upgrade #6) ───────────────
    fused = _apply_ambiguity_penalty(fused)

    # ── Step 6: Apply uncertainty floor ──────────────────────────────────────
    best_shape = max(fused, key=fused.__getitem__)
    confidence = fused[best_shape]

    if confidence < HYBRID_CONFIDENCE_FLOOR:
        rejection_reason = rejection_reason or "low_confidence"
        return PredictionResult(best_shape, confidence, False, fused, "hybrid", rejection_reason)

    accepted = confidence >= RECOG_CONFIDENCE_THRESHOLD
    if not accepted and not rejection_reason:
        rejection_reason = "below_threshold"

    return PredictionResult(best_shape, confidence, accepted, fused, "hybrid", rejection_reason)


# ─────────────────────────────────────────────────────────────────────────────
#  Unified Predictor  (Upgrade #9 — refactored, production-ready)
# ─────────────────────────────────────────────────────────────────────────────

class ShapePredictor:
    """
    Single entry-point for shape classification.
    Selects backend from config.RECOG_BACKEND:

        "rule_based" — pure geometric heuristics (no TF dependency)
        "ml"         — hybrid CNN + rule-based fusion

    Usage
    -----
        predictor = ShapePredictor()
        result = predictor.predict(stroke_points)
        if result and result.accepted:
            print(result.shape, result.confidence)
    """

    def __init__(self) -> None:
        self._backend = RECOG_BACKEND
        print(f"[predictor] Backend: '{self._backend}'  "
              f"(threshold={RECOG_CONFIDENCE_THRESHOLD}, "
              f"floor={HYBRID_CONFIDENCE_FLOOR})")
        # Pre-warm CNN if needed so first detection isn't slow
        if self._backend == "ml":
            print("[predictor] Pre-loading CNN model …")
            _load_cnn_model()

    def predict(self, points: List[Tuple[int, int]]) -> Optional[PredictionResult]:
        """
        Classify a list of (x, y) stroke points.

        Parameters
        ----------
        points : list of (x, y) int tuples — raw or decimated stroke

        Returns
        -------
        PredictionResult  – always returned if input is long enough
        None              – if input < MIN_STROKE_POINTS (too short)
        """
        from config import MIN_STROKE_POINTS
        if len(points) < MIN_STROKE_POINTS:
            return None

        if self._backend == "ml":
            result = _ml_predict(points)
        else:
            result = _rule_based_predict(points)

        # Diagnostic print for debug builds
        if result and not result.accepted:
            print(f"[predictor] Rejected: shape={result.shape}  "
                  f"conf={result.confidence:.3f}  reason={result.rejected_reason}")

        return result
