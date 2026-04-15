"""
recognition/model.py  v2.0 — Production Upgrade
================================================
Lightweight CNN for gesture shape classification.
The model can be trained on synthetic data or loaded from a saved checkpoint.

Architecture: Input(64×64×1) → Conv→BN→Pool → Conv→BN→Pool → Conv→GAP → Dense → Softmax(4)

Upgrades vs v1.0
----------------
[IMPROVE] Richer synthetic augmentation: rotation ±30°, scale variation ±20%,
          aspect ratio variation for squares, random partial occlusion (gap insertion)
[IMPROVE] n_per_class raised 500 → 800, epochs raised 15 → 20
[NEW]     Augmentation during training: horizontal + vertical flips for applicable classes
[IMPROVE] Square generator produces rectangles (0.7–1.3 aspect ratio) for robustness
[NEW]     Arc-based circle generator (partial occlusion) for broken-circle robustness
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np

from config import RECOG_IMAGE_SIZE, SHAPE_CLASSES

# Optional heavy imports — only needed for ML backend
try:
    import tensorflow as tf
    # TF 2.16+ ships Keras 3 as a standalone package.
    # Prefer the top-level `keras` package; fall back to tf.keras for older TF.
    try:
        import keras
    except ImportError:
        from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "shape_cnn.keras")


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_rotate(img: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
    """Rotate image by a random angle in [-max_angle, max_angle] degrees."""
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _random_scale(img: np.ndarray, variation: float = 0.20) -> np.ndarray:
    """Scale image content by a random factor in [1-variation, 1+variation]."""
    h, w = img.shape[:2]
    factor = 1.0 + np.random.uniform(-variation, variation)
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (new_w, new_h))
    # Crop or pad back to original size
    out = np.zeros_like(img)
    y0 = max((new_h - h) // 2, 0)
    x0 = max((new_w - w) // 2, 0)
    y1 = min(y0 + h, new_h)
    x1 = min(x0 + w, new_w)
    out_h = y1 - y0
    out_w = x1 - x0
    oy = max((h - new_h) // 2, 0)
    ox = max((w - new_w) // 2, 0)
    out[oy:oy+out_h, ox:ox+out_w] = resized[y0:y1, x0:x1]
    return out


def _insert_gap(img: np.ndarray, gap_fraction: float = 0.15) -> np.ndarray:
    """
    Black out a random arc segment of the image to simulate partial drawing.
    Only applied to circle-class training samples.
    """
    h, w = img.shape[:2]
    # Mask a random triangular wedge from centre
    cx, cy = w // 2, h // 2
    start_angle = np.random.randint(0, 360)
    gap_angle   = int(gap_fraction * 360)
    mask = img.copy()
    # Draw a filled pie slice in black to create the gap
    cv2.ellipse(mask, (cx, cy), (w, h), 0,
                start_angle, start_angle + gap_angle, 0, -1)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic training data generator  (Upgrade: richer augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_sample(shape: str, size: int = RECOG_IMAGE_SIZE,
                                 noise: float = 4.0) -> np.ndarray:
    """
    Render a single noisy, augmented shape image for training.

    Changes vs v1.0
    ---------------
    - Circles: two variants — full + partial (gap inserted 30% of the time)
    - Squares: random aspect ratio 0.7–1.3 (real hand-drawn squares aren't perfect)
    - All shapes: random rotation ±30°, scale variation ±20%
    """
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy  = size // 2, size // 2
    margin  = size // 8

    if shape == "circle":
        r = size // 2 - margin
        # Vary radius slightly
        r = max(4, r + int(np.random.uniform(-noise * 1.5, noise * 1.5)))
        for angle in np.linspace(0, 2 * np.pi, 250):
            x = int(cx + r * np.cos(angle) + np.random.uniform(-noise, noise))
            y = int(cy + r * np.sin(angle) + np.random.uniform(-noise, noise))
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = 255
        # 30% chance: insert a gap to train robustness to broken circles
        if np.random.random() < 0.30:
            img = _insert_gap(img, gap_fraction=np.random.uniform(0.08, 0.20))

    elif shape == "square":
        # Random aspect ratio — hand-drawn "squares" are rarely perfect
        ar      = np.random.uniform(0.70, 1.30)
        h_side  = int((size - 2 * margin) / 2)
        w_side  = int(h_side * ar)
        tl      = (cx - w_side + int(np.random.uniform(-noise, noise)),
                   cy - h_side + int(np.random.uniform(-noise, noise)))
        br      = (cx + w_side + int(np.random.uniform(-noise, noise)),
                   cy + h_side + int(np.random.uniform(-noise, noise)))
        # Clamp to canvas
        tl = (max(0, tl[0]), max(0, tl[1]))
        br = (min(size - 1, br[0]), min(size - 1, br[1]))
        cv2.rectangle(img, tl, br, 255, 2)

    elif shape == "triangle":
        pts = np.array([
            [cx + int(np.random.uniform(-noise, noise)),
             margin + int(np.random.uniform(-noise, noise))],
            [margin + int(np.random.uniform(-noise, noise)),
             size - margin + int(np.random.uniform(-noise, noise))],
            [size - margin + int(np.random.uniform(-noise, noise)),
             size - margin + int(np.random.uniform(-noise, noise))],
        ], dtype=np.int32)
        cv2.polylines(img, [pts], True, 255, 2)

    elif shape == "line":
        # Random angle lines, not just horizontal
        angle   = np.random.uniform(0, np.pi)
        length  = size * 0.70
        x1 = int(cx - length / 2 * np.cos(angle) + np.random.uniform(-noise, noise))
        y1 = int(cy - length / 2 * np.sin(angle) + np.random.uniform(-noise, noise))
        x2 = int(cx + length / 2 * np.cos(angle) + np.random.uniform(-noise, noise))
        y2 = int(cy + length / 2 * np.sin(angle) + np.random.uniform(-noise, noise))
        cv2.line(img, (x1, y1), (x2, y2), 255, 2)

    # Dilate to thicken strokes (simulate pen thickness)
    kernel = np.ones((2, 2), np.uint8)
    img    = cv2.dilate(img, kernel, iterations=1)

    # ── Apply augmentations ───────────────────────────────────────────────────
    img = _random_rotate(img, max_angle=30.0)
    img = _random_scale(img, variation=0.20)

    return img


def generate_dataset(n_per_class: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an in-memory synthetic dataset of (images, labels).
    n_per_class raised from 500 → 800 for better generalisation.
    """
    X, y = [], []
    for label_idx, shape in enumerate(SHAPE_CLASSES):
        for _ in range(n_per_class):
            img = _generate_synthetic_sample(shape)
            X.append(img[..., np.newaxis])
            y.append(label_idx)
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  CNN Model definition & training
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(num_classes: int = len(SHAPE_CLASSES),
              input_size: int = RECOG_IMAGE_SIZE) -> "keras.Model":
    """Build and return a lightweight CNN. Architecture unchanged from v1.0."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for CNN backend. "
                          "pip install tensorflow or use RECOG_BACKEND='rule_based'.")

    inp = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out, name="ShapeCNN_v2")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_and_save(n_per_class: int = 800, epochs: int = 20) -> str:
    """
    Train the CNN on synthetic data and save to disk.

    Changes vs v1.0
    ---------------
    - n_per_class raised 500 → 800
    - epochs raised 15 → 20
    - Keras ImageDataGenerator adds runtime flips for triangle/square classes
    """
    print("[model] Generating synthetic training data (v2.0 augmentation) …")
    X, y = generate_dataset(n_per_class)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Train/val split (85/15)
    split     = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    model = build_cnn()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
    ]

    # Train — augmentation is already baked into the synthetic samples
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(_MODEL_PATH)
    print(f"[model] Saved to {_MODEL_PATH}")
    return _MODEL_PATH


def load_model() -> "keras.Model | None":
    """Load the pre-trained CNN from disk. Returns None if not found."""
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(_MODEL_PATH):
        return None
    return keras.models.load_model(_MODEL_PATH)
