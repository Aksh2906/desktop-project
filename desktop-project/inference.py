import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque

# ---- SETTINGS ----
CONF_THRESHOLD = 0.70        # ignore weak predictions
SMOOTHING_WINDOW = 5         # frames to average

# ---- MEDIAPIPE FIRST ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---- LOAD SCALER & LABELS ----
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# ---- LAZY LOAD MODEL ----
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model("models/gesture_model.keras", compile=False)

model = None

# ---- PREDICTION SMOOTHING BUFFER ----
recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mirror camera for natural feel
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = "No Hand"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]

        # same normalization used in training
        scale = ((middle_mcp.x - wrist.x)**2 +
                 (middle_mcp.y - wrist.y)**2 +
                 (middle_mcp.z - wrist.z)**2) ** 0.5

        if scale != 0:

            # -------- BUILD 63 LANDMARK FEATURES --------
            row = []
            for lm in hand_landmarks.landmark:
                row.append((lm.x - wrist.x) / scale)
                row.append((lm.y - wrist.y) / scale)
                row.append((lm.z - wrist.z) / scale)

            row = np.array(row)

            # -------- ADD THUMB ORIENTATION --------
            wrist_vec = row[0:3]
            thumb_vec = row[12:15]
            thumb_orientation = (thumb_vec - wrist_vec) * 2

            # -------- ADD INDEX ORIENTATION --------
            index_vec = row[24:27]
            index_orientation = index_vec - wrist_vec

            # -------- FINAL FEATURE VECTOR (69 dims) --------
            row = np.concatenate([row, thumb_orientation, index_orientation])
            row = row.reshape(1, -1)

            # -------- SCALE --------
            row = scaler.transform(row)

            # -------- LOAD MODEL ONCE --------
            if model is None:
                model = load_model()

            # -------- PREDICT --------
            pred = model.predict(row, verbose=0)
            class_id = np.argmax(pred)
            confidence = float(pred[0][class_id])
            gesture = label_encoder.inverse_transform([class_id])[0]

            # ---- FIX MIRROR ISSUE (swap left/right) ----
            if gesture == "left":
                gesture = "right"
            elif gesture == "right":
                gesture = "left"

            # ---- APPLY CONFIDENCE THRESHOLD ----
            if confidence > CONF_THRESHOLD:
                recent_predictions.append(gesture)

            # ---- SMOOTH PREDICTION ----
            if len(recent_predictions) > 0:
                smoothed = max(set(recent_predictions), key=recent_predictions.count)
                prediction_text = f"{smoothed} ({confidence:.2f})"
            else:
                prediction_text = "Detecting..."

    # -------- DISPLAY --------
    cv2.putText(frame, prediction_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
