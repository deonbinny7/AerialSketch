"""
tests/run_recognition_demo.py
==============================
No-camera recognition demo — runs all 4 shape types through the
rule-based predictor and prints a pass/fail table.

Usage (from project root, with venv active):
    venv\\Scripts\\python.exe tests/run_recognition_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np


def make_circle(cx=320, cy=240, r=90, n=140):
    return [(int(cx + r * math.cos(t)), int(cy + r * math.sin(t)))
            for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]

def make_square(cx=320, cy=240, side=130, n_per_side=35):
    h = side // 2
    pts = []
    for i in range(n_per_side): pts.append((int(cx - h + side * i/n_per_side), cy - h))
    for i in range(n_per_side): pts.append((cx + h, int(cy - h + side * i/n_per_side)))
    for i in range(n_per_side): pts.append((int(cx + h - side * i/n_per_side), cy + h))
    for i in range(n_per_side): pts.append((cx - h, int(cy + h - side * i/n_per_side)))
    return pts

def make_triangle(cx=320, cy=240, size=100, n_per_side=30):
    verts = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
    pts = []
    for i in range(3):
        p0, p1 = np.array(verts[i]), np.array(verts[(i+1)%3])
        for j in range(n_per_side):
            pts.append(tuple((p0 + j/n_per_side*(p1-p0)).astype(int)))
    return pts

def make_line(n=80):
    return [(int(100 + 400 * t/n), 300 + int(5 * math.sin(t))) for t in range(n)]

def make_broken_circle(cx=320, cy=240, r=90, gap=0.35, n=80):
    arc_end = 2 * math.pi * (1 - gap)
    return [(int(cx + r * math.cos(t)), int(cy + r * math.sin(t)))
            for t in np.linspace(0, arc_end, n)]


def run():
    from recognition.predictor import _rule_based_predict
    from config import RECOG_CONFIDENCE_THRESHOLD

    test_cases = [
        ("circle",        make_circle(),           "circle"),
        ("square",        make_square(),           "square"),
        ("triangle",      make_triangle(),         "triangle"),
        ("line",          make_line(),             "line"),
        ("broken circle", make_broken_circle(),    None),   # None = expect rejection
    ]

    PASS = "\033[92m PASS \033[0m"
    FAIL = "\033[91m FAIL \033[0m"

    print("\n" + "=" * 72)
    print(f"  AerialSketch v2.0 — Shape Recognition Demo")
    print(f"  Threshold: {RECOG_CONFIDENCE_THRESHOLD}")
    print("=" * 72)
    print(f"  {'Test Case':<20} {'Predicted':<12} {'Conf':>6}  {'Accepted':<10}  Result")
    print("-" * 72)

    all_pass = True
    for name, pts, expected in test_cases:
        result = _rule_based_predict(pts)
        if result is None:
            print(f"  {name:<20} {'[None]':<12} {'N/A':>6}  {'N/A':<10}  {FAIL}")
            all_pass = False
            continue

        predicted  = result.shape
        conf       = result.confidence
        accepted   = result.accepted
        reason     = result.rejected_reason

        if expected is None:
            # Broken circle: expect rejection or very low confidence
            ok = (not accepted) or (conf < 0.80)
            label = "rejected OK" if not accepted else f"conf={conf:.2f} (penalised)"
        else:
            ok = (predicted == expected) and accepted
            label = "" if ok else f"expected={expected} reason={reason}"

        status = PASS if ok else FAIL
        if not ok:
            all_pass = False

        print(f"  {name:<20} {predicted:<12} {conf:>6.3f}  {str(accepted):<10}  {status} {label}")

    print("=" * 72)
    print(f"  Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print("=" * 72 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run())
