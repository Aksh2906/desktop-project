# GestureOS 🖐️

Control your Windows PC with hand gestures — built with MediaPipe, TensorFlow, and a browser-based UI. No cloud, no subscription, runs fully local.

---

## What it does

- Live webcam gesture detection using MediaPipe hand landmarks
- Bind any gesture to a Windows action — volume, brightness, media keys, open apps, screenshots, custom shell commands
- **Volume and brightness adjust continuously by pinching fingers in/out** (no button press needed)
- Train new gestures directly from the browser — record → train → use
- Delete a gesture and the model automatically retrains without it

---

## Project Structure

```
GestureOS/
│
├── index.html            ← Browser UI (camera, gesture library, training)
├── server.py             ← WebSocket server (MediaPipe + inference + OS actions)
├── controls.py           ← All Windows OS control functions + pinch logic
├── train_model.py        ← Neural network training script
│
├── data/                 ← Auto-created — one CSV per gesture
│   ├── open_palm.csv
│   └── closed_fist.csv
│
├── gesture_model.keras   ← Auto-created after first training
├── scaler.pkl            ← Auto-created after first training
├── label_encoder.pkl     ← Auto-created after first training
│
├── env/                  ← Your virtual environment (not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

- **Windows 10 or 11**
- **Python 3.9, 3.10, or 3.11** — TensorFlow does not support 3.12 yet
- A webcam
- Chrome or Edge (camera requires `localhost`, not `file://`)

---

## Setup

### 1 — Clone the repo

```bash
git clone https://github.com/your-username/GestureOS.git
cd GestureOS
```

### 2 — Create a virtual environment

```bash
python -m venv env
```

Activate it:

```bash
# CMD
env\Scripts\activate.bat

# PowerShell (if you get an execution policy error, run this first once)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
env\Scripts\Activate.ps1
```

You should see `(env)` at the start of your terminal prompt.

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

TensorFlow is large — this may take a few minutes on first install.

### 4 — Start the WebSocket server

```bash
python server.py
```

Expected output:
```
Server running at ws://127.0.0.1:8765
```

Keep this terminal open.

### 5 — Serve the frontend

Open a **second terminal**, activate env, then:

```bash
python -m http.server 8080
```

Open **http://localhost:8080** in Chrome or Edge.

> ⚠️ Never open `index.html` directly via `file://` — the browser blocks camera access on file:// URLs.

---

## Usage

### Training a gesture

1. Click **+ Add Gesture**
2. Enter a name — use `lowercase_with_underscores` (e.g. `open_palm`, `closed_fist`)
3. Choose a control (Volume Up, Play/Pause, Open Webpage, etc.)
4. Click **Save Gesture** — the modal stays open
5. Click **● Start Recording**, hold your hand gesture steady, wait for progress bar to fill (default 150 frames)
6. Click **⚡ Train Now**
7. Repeat for each gesture — **minimum 2 gestures needed** before training works

### Pinch-controlled volume and brightness

When a gesture is bound to **Volume Up/Down** or **Brightness Up/Down**, holding that gesture activates continuous pinch mode:

| Hand movement | Effect |
|--------------|--------|
| Spread thumb + index apart | Increase value |
| Pinch thumb + index together | Decrease value |
| Hold still | No change (dead zone) |

The change is smooth and proportional to how wide you spread.

### Deleting a gesture

Click 🗑 next to any gesture. This:
- Removes it from the UI
- Deletes its training CSV from `data/`
- Automatically retrains the model — the gesture will no longer be recognised

---

## How the ML pipeline works

```
Webcam frame (browser)
        ↓  WebSocket (base64 JPEG)
server.py receives frame
        ↓
MediaPipe Hands — 21 landmarks (x, y, z)
        ↓
Normalise: subtract wrist, scale by wrist→MCP9 distance → 63 values
Add orientation features (thumb + index direction)        → 69 values
        ↓
StandardScaler  (scaler.pkl)
        ↓
Dense neural network  (gesture_model.keras)
  Input(69) → Dense(256) → BN → Dropout → Dense(128) → BN → Dropout → Softmax(N)
        ↓
Smoothing: majority vote over last 5 frames
        ↓
confidence ≥ 70%  →  controls.py fires the OS action
```

---

## File descriptions

| File | Role |
|------|------|
| `server.py` | WebSocket server at `ws://127.0.0.1:8765`. Receives frames, runs MediaPipe, runs model, sends predictions. Handles recording, training, deletion. |
| `controls.py` | All Windows OS actions. `execute_control(type, param)` for one-shot, `update_pinch(landmarks, mode)` for continuous pinch. |
| `train_model.py` | Reads CSVs from `data/`, trains the network, saves model + scaler + label encoder. Called by `server.py` automatically. |
| `index.html` | Single-file frontend — no build step needed. Manages gesture library in `localStorage`, streams frames over WebSocket. |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: pycaw` | `pip install pycaw comtypes` |
| `ModuleNotFoundError: screen_brightness_control` | `pip install screen-brightness-control` |
| Brightness not working | Some laptop displays don't expose software brightness control. Try using `Custom Command` with `nircmd.exe` instead |
| Camera not starting | Use `http://localhost:8080`, never `file://` |
| "No Model — Train first" | Record data for at least 2 gestures, click Train |
| Gesture names not matching | Name in UI must exactly match the CSV label — check Model Status panel |
| Low accuracy | Record 200+ frames, good consistent lighting, make gestures visually very different from each other |
| PowerShell execution policy error | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## Tips for good accuracy

- Record **150–200+ frames** per gesture
- Use **consistent, even lighting** — avoid windows behind you
- While recording, make **small natural variations** (slight rotation, distance changes) — don't hold completely rigid
- Make gestures **visually distinct** — two similar-looking gestures will confuse the model
- If accuracy is low, delete and re-record the problem gesture with more frames
