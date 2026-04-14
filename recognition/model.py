"""
recognition/model.py
====================
Lightweight CNN for gesture shape classification.
The model can be trained on synthetic data or loaded from a saved checkpoint.

Architecture: Input(64×64×1) → Conv→Pool → Conv→Pool → Dense → Softmax(4)
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
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "shape_cnn.keras")


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic training data generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_sample(shape: str, size: int = RECOG_IMAGE_SIZE,
                                 noise: float = 5.0) -> np.ndarray:
    """Render a single noisy shape image for training."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    margin = size // 8

    if shape == "circle":
        r = size // 2 - margin
        for angle in np.linspace(0, 2 * np.pi, 200):
            x = int(cx + r * np.cos(angle) + np.random.uniform(-noise, noise))
            y = int(cy + r * np.sin(angle) + np.random.uniform(-noise, noise))
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = 255

    elif shape == "square":
        top_left     = (margin + int(np.random.uniform(-noise, noise)),
                        margin + int(np.random.uniform(-noise, noise)))
        bottom_right = (size - margin + int(np.random.uniform(-noise, noise)),
                        size - margin + int(np.random.uniform(-noise, noise)))
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

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
        x1 = margin + int(np.random.uniform(-noise, noise))
        y1 = int(size * np.random.uniform(0.3, 0.7))
        x2 = size - margin + int(np.random.uniform(-noise, noise))
        y2 = int(size * np.random.uniform(0.3, 0.7))
        cv2.line(img, (x1, y1), (x2, y2), 255, 2)

    # Dilate slightly to thicken strokes
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img


def generate_dataset(n_per_class: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Build an in-memory synthetic dataset of (images, labels)."""
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
    """Build and return a lightweight CNN."""
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

    model = keras.Model(inp, out, name="ShapeCNN")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_and_save(n_per_class: int = 500, epochs: int = 15) -> str:
    """
    Train the CNN on synthetic data and save to disk.
    Returns the saved model path.
    """
    print("[model] Generating synthetic training data …")
    X, y = generate_dataset(n_per_class)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Train/val split
    split = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    model = build_cnn()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=epochs,
              batch_size=32,
              callbacks=callbacks,
              verbose=1)

    model.save(_MODEL_PATH)
    print(f"[model] Saved → {_MODEL_PATH}")
    return _MODEL_PATH


def load_model() -> "keras.Model | None":
    """Load the pre-trained CNN from disk. Returns None if not found."""
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(_MODEL_PATH):
        return None
    return keras.models.load_model(_MODEL_PATH)
