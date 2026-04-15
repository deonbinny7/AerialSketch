import sys, math
sys.path.insert(0, '.')
import numpy as np
from recognition.predictor import _rule_based_predict
from config import RECOG_CONFIDENCE_THRESHOLD

def make_circle(cx=320, cy=240, r=90, n=140):
    return [(int(cx + r*math.cos(t)), int(cy + r*math.sin(t)))
            for t in np.linspace(0, 2*math.pi, n, endpoint=False)]

def make_square(cx=320, cy=240, side=130, n_per_side=35):
    h = side//2; pts = []
    for i in range(n_per_side): pts.append((int(cx-h+side*i/n_per_side), cy-h))
    for i in range(n_per_side): pts.append((cx+h, int(cy-h+side*i/n_per_side)))
    for i in range(n_per_side): pts.append((int(cx+h-side*i/n_per_side), cy+h))
    for i in range(n_per_side): pts.append((cx-h, int(cy+h-side*i/n_per_side)))
    return pts

def make_triangle(cx=320, cy=240, size=100, n_per_side=30):
    verts = [(cx,cy-size),(cx-size,cy+size),(cx+size,cy+size)]
    pts = []
    for i in range(3):
        p0, p1 = np.array(verts[i]), np.array(verts[(i+1)%3])
        for j in range(n_per_side):
            pts.append(tuple((p0 + j/n_per_side*(p1-p0)).astype(int)))
    return pts

def make_line(n=80):
    return [(int(100+400*t/n), 300+int(5*math.sin(t))) for t in range(n)]

def make_broken_circle(gap=0.35, n=80):
    arc = 2*math.pi*(1-gap)
    return [(int(320+90*math.cos(t)),int(240+90*math.sin(t)))
            for t in np.linspace(0, arc, n)]

cases = [
    ('circle',       make_circle(),      'circle'),
    ('square',       make_square(),      'square'),
    ('triangle',     make_triangle(),    'triangle'),
    ('line',         make_line(),        'line'),
    ('broken_circle',make_broken_circle(), None),
]

for name, pts, exp in cases:
    r = _rule_based_predict(pts)
    print(f'{name}: shape={r.shape} conf={r.confidence:.3f} accepted={r.accepted} reason={repr(r.rejected_reason)}')
    if exp is not None:
        ok = (r.shape == exp) and r.accepted
        print(f'  => {"PASS" if ok else "FAIL"} (expected={exp})')
        if not ok:
            print(f'     all_scores={r.all_scores}')
    else:
        ok = (not r.accepted) or (r.confidence < 0.80)
        print(f'  => {"PASS" if ok else "FAIL"} (broken circle partial penalty check)')
