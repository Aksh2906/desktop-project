import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import os
import joblib
import subprocess
import pandas as pd
from pathlib import Path
from collections import deque
import mediapipe as mp
import tensorflow as tf

from controls import execute_control, update_pinch, reset_pinch, PINCH_CONTROLS

# ================= CONFIG =================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH  = BASE_DIR / "gesture_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
LABEL_PATH  = BASE_DIR / "label_encoder.pkl"

HOST = "127.0.0.1"
PORT = 8765

CONF_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5

# ================= GLOBALS =================
model = None
scaler = None
label_encoder = None
recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

recording = False
record_name = None
record_target = 150
record_buffer = []

# gesture_name → pinch mode ("volume" | "brightness")
# Browser sends update_pinch_map whenever user saves/deletes a gesture
active_pinch_map: dict[str, str] = {}

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ================= LOAD =================
def load_artifacts():
    global scaler, label_encoder
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
    if LABEL_PATH.exists():
        label_encoder = joblib.load(LABEL_PATH)

def load_model():
    global model
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ================= FEATURES =================
def extract_features(frame):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0]
    wrist = lm.landmark[0]
    middle = lm.landmark[9]

    scale = ((middle.x - wrist.x)**2 +
             (middle.y - wrist.y)**2 +
             (middle.z - wrist.z)**2) ** 0.5

    if scale == 0:
        return None

    row = []
    for p in lm.landmark:
        row.append((p.x - wrist.x)/scale)
        row.append((p.y - wrist.y)/scale)
        row.append((p.z - wrist.z)/scale)

    row = np.array(row)

    wrist_vec = row[0:3]
    thumb_vec = row[12:15]
    index_vec = row[24:27]

    thumb_orientation = (thumb_vec - wrist_vec) * 2.0
    index_orientation = index_vec - wrist_vec

    feature69 = np.concatenate([row, thumb_orientation, index_orientation])

    # Also return the raw landmarks object for pinch detection
    return row, feature69, lm

# ================= SAVE CSV =================
def save_recording():
    global record_buffer, record_name

    if not record_buffer:
        print("No data recorded")
        return 0

    df = pd.DataFrame(record_buffer)
    df["label"] = record_name

    file_path = DATA_DIR / f"{record_name}.csv"

    if file_path.exists():
        old = pd.read_csv(file_path)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(file_path, index=False)
    saved = len(record_buffer)
    print(f"[REC] Saved {saved} samples → {file_path}")
    record_buffer = []
    return saved

# ================= PREDICT =================
def predict(features):
    if model is None or scaler is None or label_encoder is None:
        return None, 0.0

    x = scaler.transform(features.reshape(1,-1))
    pred = model.predict(x, verbose=0)
    cid = np.argmax(pred)
    conf = float(pred[0][cid])
    gesture = label_encoder.inverse_transform([cid])[0]
    return gesture, conf

# ================= WEBSOCKET =================
async def handler(ws):
    global recording, record_name, record_buffer, model, scaler, label_encoder, active_pinch_map

    await ws.send(json.dumps({
        "type": "status",
        "model_loaded": model is not None,
        "classes": list(label_encoder.classes_) if label_encoder else []
    }))

    async for raw in ws:
        msg = json.loads(raw)

        # ── VIDEO FRAME ────────────────────────────────────────────────────
        if msg["type"] == "frame":
            data = msg["data"].split(",")[1]
            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(data), np.uint8), 1
            )

            result = extract_features(frame)
            if result is None:
                recent_predictions.clear()
                reset_pinch()
                await ws.send(json.dumps({"type":"prediction","hand":False}))
                continue

            row63, feat69, raw_landmarks = result

            # Recording: append landmark row and track progress
            if recording:
                record_buffer.append(row63)
                count  = len(record_buffer)
                target = record_target

                await ws.send(json.dumps({
                    "type":   "record_progress",
                    "name":   record_name,
                    "count":  count,
                    "target": target,
                }))

                if count >= target:
                    recording = False
                    frames_saved = save_recording()
                    await ws.send(json.dumps({
                        "type":   "record_done",
                        "name":   record_name,
                        "frames": frames_saved,
                    }))
                continue  # don't predict while recording

            # Inference
            gesture, conf = predict(feat69)

            if gesture and conf > CONF_THRESHOLD:
                recent_predictions.append(gesture)

            smoothed = max(set(recent_predictions),
                           key=recent_predictions.count) if recent_predictions else None

            # ── Pinch control (continuous) ──────────────────────────────────
            # The browser keeps a gesture→control_type mapping in localStorage.
            # It sends execute_control for one-shots (handled below).
            # For pinch-enabled controls it sends a "pinch_frame" message
            # with the control_type so server can call update_pinch() here.
            pinch_result = {}
            if smoothed and smoothed in active_pinch_map:
                mode = active_pinch_map[smoothed]
                pinch_result = update_pinch(raw_landmarks, mode)
            else:
                reset_pinch()

            response = {
                "type":       "prediction",
                "hand":       True,
                "gesture":    gesture,
                "smoothed":   smoothed,
                "confidence": round(conf*100, 1),
                "hand_x":     0.0,
                "hand_y":     0.0,
            }
            if pinch_result.get("changed"):
                response["pinch"] = {
                    "mode":  pinch_result["mode"],
                    "value": pinch_result["value"],
                }

            await ws.send(json.dumps(response))

        # ── UPDATE PINCH MAP (sent by browser on gesture save/delete) ──────
        elif msg["type"] == "update_pinch_map":
            # msg["map"] = {"open_palm": "volume", "thumbs_up": "brightness", ...}
            active_pinch_map = msg.get("map", {})
            print(f"[PINCH] Map updated: {active_pinch_map}")

        # ── ONE-SHOT OS CONTROL (from browser for non-pinch gestures) ──────
        elif msg["type"] == "execute_control":
            execute_control(msg.get("control_type",""), msg.get("param",""))

        # ── START RECORDING ────────────────────────────────────────────────
        elif msg["type"] == "record_start":
            recording     = True
            record_name   = msg["gesture_name"]
            record_target = int(msg.get("frame_count", 150))
            record_buffer = []
            print(f"[REC] Started: '{record_name}' target={record_target}")

        # ── STOP RECORDING (manual early stop) ────────────────────────────
        elif msg["type"] == "record_stop":
            recording = False
            frames_saved = save_recording()
            print(f"[REC] Stopped early: {frames_saved} frames for '{record_name}'")
            await ws.send(json.dumps({
                "type":   "record_stopped",
                "name":   record_name,
                "frames": frames_saved,
            }))

        # ── TRAIN ──────────────────────────────────────────────────────────
        elif msg["type"] == "train_model":
            print("[TRAIN] Starting train_model.py …")
            await ws.send(json.dumps({"type": "train_started"}))
            try:
                result = subprocess.run(
                    ["python", "train_model.py"],
                    capture_output=True, text=True
                )
                # Parse final accuracy from stdout
                accuracy = 0.0
                for line in result.stdout.splitlines():
                    if "Final Accuracy:" in line:
                        try: accuracy = round(float(line.split(":")[-1].strip())*100, 1)
                        except: pass

                load_artifacts()
                load_model()
                classes = list(label_encoder.classes_) if label_encoder else []
                print(f"[TRAIN] Done. Accuracy={accuracy}%  Classes={classes}")
                await ws.send(json.dumps({
                    "type":     "train_done",
                    "accuracy": accuracy,
                    "classes":  classes,
                }))
            except Exception as e:
                await ws.send(json.dumps({"type":"train_error","message":str(e)}))

        # ── GET DATA INFO (frame counts per gesture CSV) ──────────────────
        elif msg["type"] == "get_data_info":
            data = {}
            for csv_path in DATA_DIR.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_path)
                    data[csv_path.stem] = len(df)
                except:
                    data[csv_path.stem] = 0
            await ws.send(json.dumps({"type":"data_info","data":data}))

        # ── DELETE GESTURE DATA + RETRAIN ─────────────────────────────────
        elif msg["type"] == "delete_and_retrain":
            name = msg.get("gesture_name", "").strip()
            csv_path = DATA_DIR / f"{name}.csv"

            # Delete the CSV so the gesture is gone from training data
            if csv_path.exists():
                csv_path.unlink()
                print(f"[DEL] Deleted training data for '{name}'")
            else:
                print(f"[DEL] No CSV found for '{name}' (nothing to delete)")

            # Count remaining CSVs
            remaining = list(DATA_DIR.glob("*.csv"))

            if len(remaining) < 2:
                # Not enough classes to retrain — clear model files so it stops predicting
                for f in [MODEL_PATH, SCALER_PATH, LABEL_PATH]:
                    if f.exists():
                        f.unlink()
                model = None; scaler = None; label_encoder = None
                print(f"[DEL] Only {len(remaining)} class(es) left — model cleared")
                await ws.send(json.dumps({
                    "type":     "train_done",
                    "accuracy": 0.0,
                    "classes":  [],
                    "message":  f"'{name}' deleted. Need at least 2 gestures to train a model.",
                }))
            else:
                # Retrain without the deleted gesture
                print(f"[DEL] Retraining without '{name}' …")
                await ws.send(json.dumps({"type": "train_started"}))
                try:
                    result = subprocess.run(
                        ["python", "train_model.py"],
                        capture_output=True, text=True
                    )
                    accuracy = 0.0
                    for line in result.stdout.splitlines():
                        if "Final Accuracy:" in line:
                            try: accuracy = round(float(line.split(":")[-1].strip())*100, 1)
                            except: pass

                    load_artifacts()
                    load_model()
                    classes = list(label_encoder.classes_) if label_encoder else []
                    print(f"[DEL] Retrain done. Classes={classes}  Accuracy={accuracy}%")
                    await ws.send(json.dumps({
                        "type":     "train_done",
                        "accuracy": accuracy,
                        "classes":  classes,
                    }))
                except Exception as e:
                    print(f"[DEL] Retrain failed: {e}")
                    await ws.send(json.dumps({"type":"train_error","message":str(e)}))

# ================= MAIN =================
async def main():
    load_artifacts()
    load_model()
    print("Server running at ws://127.0.0.1:8765")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
