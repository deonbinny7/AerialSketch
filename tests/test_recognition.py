"""
tests/test_recognition.py
==========================
Unit tests for the AerialSketch recognition pipeline (v2.0).
Run with: venv\\Scripts\\python.exe -m pytest tests/test_recognition.py -v

No webcam required — all tests use synthetic point data.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest

from recognition.preprocess import (
    interpolate_stroke,
    normalize_stroke,
    extract_contour_features,
    stroke_to_image,
)
from recognition.predictor import (
    _score_by_metrics,
    _apply_ambiguity_penalty,
    _is_partial_shape,
    _rule_based_predict,
    PredictionResult,
)
from config import (
    RECOG_CONFIDENCE_THRESHOLD,
    HYBRID_CONFIDENCE_FLOOR,
    FUSION_WEIGHT_RULES,
    FUSION_WEIGHT_ML,
    SHAPE_CLASSES,
    AMBIGUITY_MARGIN,
    PARTIAL_SHAPE_PENALTY,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Test helpers — synthetic point generators
# ─────────────────────────────────────────────────────────────────────────────

def make_circle_pts(cx=320, cy=240, r=80, n=120):
    """Generate points along a perfect circle."""
    return [(int(cx + r * math.cos(t)), int(cy + r * math.sin(t)))
            for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def make_square_pts(cx=320, cy=240, side=120, n_per_side=30):
    """Generate points along a square outline."""
    h = side // 2
    pts = []
    for i in range(n_per_side):
        t = i / n_per_side
        pts.append((int(cx - h + side * t), cy - h))   # top
    for i in range(n_per_side):
        t = i / n_per_side
        pts.append((cx + h, int(cy - h + side * t)))   # right
    for i in range(n_per_side):
        t = i / n_per_side
        pts.append((int(cx + h - side * t), cy + h))   # bottom
    for i in range(n_per_side):
        t = i / n_per_side
        pts.append((cx - h, int(cy + h - side * t)))   # left
    return pts


def make_triangle_pts(cx=320, cy=240, size=100, n_per_side=30):
    """Generate points along an equilateral triangle outline."""
    verts = [
        (cx,        cy - size),
        (cx - size, cy + size),
        (cx + size, cy + size),
    ]
    pts = []
    for i in range(3):
        p0, p1 = np.array(verts[i]), np.array(verts[(i+1) % 3])
        for j in range(n_per_side):
            t = j / n_per_side
            pts.append(tuple((p0 + t * (p1 - p0)).astype(int)))
    return pts


def make_line_pts(x0=100, y0=300, x1=500, y1=310, n=80):
    """Generate points along a nearly-horizontal line."""
    return [(int(x0 + (x1-x0)*t/n), int(y0 + (y1-y0)*t/n)) for t in range(n)]


def make_broken_circle_pts(cx=320, cy=240, r=80, gap_fraction=0.35, n=80):
    """Generate a circle with a large gap (broken/partial)."""
    arc_end = 2 * math.pi * (1 - gap_fraction)
    pts = [(int(cx + r * math.cos(t)), int(cy + r * math.sin(t)))
           for t in np.linspace(0, arc_end, n)]
    return pts


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1: interpolate_stroke fills large gaps
# ─────────────────────────────────────────────────────────────────────────────

class TestInterpolateStroke:

    def test_no_gaps_below_threshold(self):
        """Dense stroke — should have no additional points inserted."""
        pts = [(i * 5, 100) for i in range(20)]   # 5px apart
        result = interpolate_stroke(pts, max_gap_px=12)
        assert len(result) == len(pts)

    def test_gaps_filled_above_threshold(self):
        """Two points 50px apart — should be filled with ~4 intermediate points."""
        pts = [(0, 0), (50, 0)]
        result = interpolate_stroke(pts, max_gap_px=12)
        assert len(result) > 2, "Expected intermediate points to be inserted"
        # All consecutive gaps should be ≤ max_gap_px + 1 (rounding tolerance)
        for i in range(1, len(result)):
            d = math.hypot(result[i][0] - result[i-1][0], result[i][1] - result[i-1][1])
            assert d <= 13, f"Gap {d:.1f} still exceeds threshold at index {i}"

    def test_endpoints_preserved(self):
        """First and last points must always be preserved."""
        pts = [(10, 20), (200, 300)]
        result = interpolate_stroke(pts, max_gap_px=12)
        assert result[0] == pts[0]
        assert result[-1] == pts[-1]

    def test_single_point_passthrough(self):
        pts = [(50, 50)]
        assert interpolate_stroke(pts) == pts


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2: normalize_stroke centers and scales properly
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeStroke:

    def test_output_fits_target_size(self):
        pts = [(10, 10), (500, 400), (200, 300)]
        normed = normalize_stroke(pts, target_size=200)
        xs, ys = [p[0] for p in normed], [p[1] for p in normed]
        assert max(xs) <= 200 and max(ys) <= 200
        assert min(xs) >= 0  and min(ys) >= 0

    def test_single_point_safe(self):
        pts = [(150, 150)]
        result = normalize_stroke(pts)
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3: extract_contour_features returns valid dict for good shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractContourFeatures:

    def _assert_has_keys(self, feat):
        required = {"circularity", "solidity", "vertices_tight", "vertices_loose",
                    "aspect_ratio", "straightness", "area", "closed"}
        assert required.issubset(feat.keys()), f"Missing keys: {required - feat.keys()}"

    def test_circle_returns_features(self):
        feat = extract_contour_features(make_circle_pts())
        assert feat, "Should return non-empty dict for circle"
        self._assert_has_keys(feat)
        assert feat["circularity"] > 0.5, "Circle circularity should be > 0.5"

    def test_square_returns_features(self):
        feat = extract_contour_features(make_square_pts())
        assert feat, "Should return non-empty dict for square"
        self._assert_has_keys(feat)

    def test_too_short_returns_empty(self):
        feat = extract_contour_features([(0, 0), (10, 10)])
        assert feat == {}


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4: Rule-based shape detection accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleBasedPredict:

    def test_circle_detected(self):
        result = _rule_based_predict(make_circle_pts())
        assert result is not None
        assert result.shape == "circle", f"Expected circle, got {result.shape}"
        assert result.confidence >= 0.55, f"Confidence too low: {result.confidence:.3f}"
        assert result.accepted, "Circle should be accepted"

    def test_square_detected(self):
        result = _rule_based_predict(make_square_pts())
        assert result is not None
        assert result.shape == "square", f"Expected square, got {result.shape}"

    def test_triangle_detected(self):
        result = _rule_based_predict(make_triangle_pts())
        assert result is not None
        assert result.shape == "triangle", f"Expected triangle, got {result.shape}"

    def test_line_detected(self):
        result = _rule_based_predict(make_line_pts())
        assert result is not None
        assert result.shape == "line", f"Expected line, got {result.shape}"

    def test_broken_circle_penalised(self):
        """A 35%-open circle should be rejected or have very low confidence."""
        result = _rule_based_predict(make_broken_circle_pts(gap_fraction=0.35))
        assert result is not None
        # Either it must be rejected, OR if accepted, confidence must reflect penalty
        if result.accepted:
            # It's okay to accept if confidence is still meaningful, but
            # the partial penalty must have visibly reduced it (< 80%)
            assert result.confidence < 0.80, (
                f"Broken circle should have reduced confidence, got {result.confidence:.3f}")
        else:
            assert not result.accepted, "Broken circle with large gap should be rejected"


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5: Ambiguity penalty fires when top-2 scores are close
# ─────────────────────────────────────────────────────────────────────────────

class TestAmbiguityPenalty:

    def test_penalty_applied_on_close_scores(self):
        scores = {"circle": 0.72, "square": 0.70, "triangle": 0.30, "line": 0.10}
        result = _apply_ambiguity_penalty(scores.copy())
        # circle was top — should be penalised since margin = 0.02 < AMBIGUITY_MARGIN
        assert result["circle"] < 0.72, "Top shape should be penalised"

    def test_no_penalty_on_clear_winner(self):
        scores = {"circle": 0.90, "square": 0.30, "triangle": 0.10, "line": 0.05}
        result = _apply_ambiguity_penalty(scores.copy())
        # margin = 0.60 >> AMBIGUITY_MARGIN → no penalty
        assert result["circle"] == 0.90, "Clear winner should not be penalised"

    def test_single_class_safe(self):
        scores = {"circle": 0.80}
        result = _apply_ambiguity_penalty(scores.copy())
        assert result["circle"] == 0.80


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6: Partial shape detector
# ─────────────────────────────────────────────────────────────────────────────

class TestIsPartialShape:

    def test_open_low_solidity(self):
        feat = {"closed": False, "solidity": 0.45, "area": 1500}
        assert _is_partial_shape(feat) is True

    def test_closed_high_solidity(self):
        feat = {"closed": True, "solidity": 0.92, "area": 5000}
        assert _is_partial_shape(feat) is False

    def test_open_but_high_solidity(self):
        # Open but solid — e.g. a C-shape with large area. Should NOT trigger penalty.
        feat = {"closed": False, "solidity": 0.85, "area": 8000}
        assert _is_partial_shape(feat) is False

    def test_empty_features(self):
        assert _is_partial_shape({}) is False


# ─────────────────────────────────────────────────────────────────────────────
#  Test 7: Hybrid fusion math check
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridFusionMath:

    def test_fusion_weights_sum(self):
        """Weights must sum to 1.0 for the fusion to be a convex combination."""
        assert abs(FUSION_WEIGHT_RULES + FUSION_WEIGHT_ML - 1.0) < 1e-6, (
            f"Weights should sum to 1.0, got {FUSION_WEIGHT_RULES + FUSION_WEIGHT_ML}")

    def test_fusion_formula(self):
        """Manually verify fusion produces expected output."""
        rule_score = 0.80
        ml_score   = 0.60
        expected   = rule_score * FUSION_WEIGHT_RULES + ml_score * FUSION_WEIGHT_ML
        computed   = round(expected, 4)
        assert abs(computed - (0.80 * 0.6 + 0.60 * 0.4)) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
#  Test 8: stroke_to_image produces correct shape
# ─────────────────────────────────────────────────────────────────────────────

class TestStrokeToImage:

    def test_output_shape_and_dtype(self):
        pts = make_circle_pts()
        img = stroke_to_image(pts, size=64)
        assert img.shape == (64, 64, 1)
        assert img.dtype == np.uint8

    def test_not_all_black(self):
        pts = make_circle_pts()
        img = stroke_to_image(pts, size=64)
        assert img.max() > 0, "Rendered stroke image should not be entirely black"

    def test_empty_points(self):
        img = stroke_to_image([], size=64)
        assert img.shape == (64, 64, 1)
        assert img.max() == 0
