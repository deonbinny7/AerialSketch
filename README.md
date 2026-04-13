# ✦ AerialSketch

> **AI-Powered Gesture Drawing System** — Draw in the air with your finger, have shapes recognised in real-time, and receive instant voice feedback.

---

## 🎯 Project Description

AerialSketch is a production-grade computer vision application that transforms your webcam into an intelligent drawing surface. Using **MediaPipe** for hand tracking, a **rule-based geometric recogniser** (with optional CNN upgrade), and **offline TTS voice synthesis**, the system detects when you're drawing, classifies your shape, and announces it — all at 30-60+ FPS.

---

## ✨ Features

| Feature | Details |
|---|---|
| ✋ Hand tracking | MediaPipe Hands — 21 landmarks, sub-30ms |
| 🎨 Multi-color canvas | 4 palette colors, cycle with `C` key |
| 🖌️ Adjustable brush | 4 sizes, cycle with `B` key |
| 🧠 Shape recognition | Circle · Square · Triangle · Line |
| 🔊 Voice feedback | Offline pyttsx3 TTS on background thread |
| ↩️ Undo / Redo | 20-level history (`Z` / `Y`) |
| 💾 Save drawing | PNG export with timestamp (`S`) |
| 🪄 Shape beautify | Rough sketch replaced by clean vector |
| 🌑 Dark theme UI | Glassmorphic toolbar, glow cursor |
| ⚡ Eraser mode | Toggle with `E` |
| 🐛 Debug mode | Toggle with `D` |

---

## 🧠 Tech Stack

- **Python 3.10+**
- **OpenCV** `4.9` — rendering, video I/O
- **MediaPipe** `0.10` — hand landmark detection
- **NumPy** `1.26` — canvas & geometry
- **pyttsx3** `2.90` — offline TTS
- *(Optional)* **TensorFlow** `2.16` — CNN classifier

---

## 📁 Project Structure

```
AerialSketch/
├── main.py                   # Application entry point
├── config.py                 # All constants (no hardcoding elsewhere)
├── requirements.txt
│
├── hand_tracking/
│   └── tracker.py            # MediaPipe wrapper + gesture state machine
│
├── drawing/
│   └── canvas.py             # NumPy canvas, stroke smoother, undo/redo
│
├── recognition/
│   ├── preprocess.py         # Stroke → normalised image + geometric features
│   ├── model.py              # CNN architecture + synthetic data generator
│   └── predictor.py          # Unified predict() API (rule-based or ML)
│
├── audio/
│   └── voice.py              # Non-blocking TTS on daemon thread
│
├── ui/
│   └── interface.py          # Toolbar, glow cursor, badge, toast rendering
│
├── utils/
│   └── helpers.py            # FPS counter, glow, text, save utilities
│
└── assets/
    └── sounds/               # (Optional custom SFX)
```

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/deonbinny7/AerialSketch.git
cd AerialSketch

# 2. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install TensorFlow for CNN backend
pip install tensorflow==2.16.1
```

---

## ▶️ Usage

```bash
python main.py
```

### Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger up | **Draw** |
| ✌️ Two fingers up | **Pause** (lift pen) |
| 🖐️ Open palm (hold) | **Clear canvas** |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `C` | Cycle through 4 colors |
| `B` | Cycle brush sizes |
| `Z` | Undo |
| `Y` | Redo |
| `S` | Save canvas as PNG |
| `E` | Toggle eraser |
| `D` | Toggle debug mode |
| `Q` / `ESC` | Quit |

---

## ⚙️ Configuration

All settings live in `config.py` — camera resolution, recognition thresholds, color palette, TTS rate, brush sizes, and more. **Never hardcode values in module files.**

To switch to CNN recognition:

```python
# config.py
RECOG_BACKEND: str = "ml"
```

Then run once to auto-train the CNN:

```bash
python -c "from recognition.model import train_and_save; train_and_save()"
```

---

## 📖 Module Explanations

### `hand_tracking/tracker.py`
Wraps MediaPipe Hands. Extracts landmark 8 (index fingertip) every frame. Uses a **debounce buffer** to require N consecutive frames before switching gesture state — preventing false triggers from finger jitter.

### `drawing/canvas.py`
Maintains a persistent NumPy BGR canvas. Points from the tracker are smoothed via a **moving average filter** before being drawn with `cv2.line`. Undo/redo is implemented via canvas snapshots in a bounded deque.

### `recognition/preprocess.py`
Converts raw stroke points to a normalised 64×64 binary image (for CNN) and extracts geometric features like **circularity**, **straightness**, **corner count**, and **aspect ratio** (for rule-based mode).

### `recognition/predictor.py`
Factory that selects the backend from config. Rule-based mode scores each shape against the geometric features. CNN mode runs a forward pass through the trained model. Both return a `PredictionResult` with `shape`, `confidence`, and `accepted` flag.

### `audio/voice.py`
Runs `pyttsx3` on a **daemon thread**, draining a bounded queue. The main loop is never blocked by speech. Excess announcements are dropped to prevent audio lag.

### `ui/interface.py`
Pure OpenCV rendering. The toolbar uses `addWeighted` for glassmorphism. The glow cursor layers multiple blurred circles. The shape badge fades out gracefully over 3.5 seconds.

---

## 🔮 Future Improvements

- [ ] Multi-hand support (left = palette, right = draw)
- [ ] Gesture-based undo (swipe)
- [ ] WebRTC streaming for browser-based UI
- [ ] Export as SVG
- [ ] Custom shape training from live data
- [ ] Digit recognition (0–9)

---

## 🎥 Demo Recording Script

1. **Open** the app — show the dark UI and toolbar
2. **Draw a circle** slowly with your index finger — show glow cursor
3. **Lift finger** — show badge appear + speak "Circle detected"
4. **Draw a line** — show badge + voice
5. **Press C** to change color, draw a triangle
6. **Press Z** to undo
7. **Open palm** to clear canvas
8. **Press S** to save, show saved file

---

## 📄 License

MIT © 2024 AerialSketch
