import os
import cv2
import csv
import mediapipe as mp

# ---------------- SETTINGS ----------------
FOLDERS = {
    "up": "UP/extracted_frames_UP_s6"
    
}

OUTPUT_CSV = "up_s5.csv"



# ------------------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

dataset = []
skipped_images = 0

print("Starting CSV creation...\n")

for label, folder in FOLDERS.items():

    print(f"Processing {label.upper()} images...")

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        image = cv2.imread(img_path)
        if image is None:
            skipped_images += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            skipped_images += 1
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]

        scale = ((middle_mcp.x - wrist.x)**2 +
                 (middle_mcp.y - wrist.y)**2 +
                 (middle_mcp.z - wrist.z)**2) ** 0.5

        if scale == 0:
            skipped_images += 1
            continue

        row = []

        for lm in hand_landmarks.landmark:
            row.append((lm.x - wrist.x) / scale)
            row.append((lm.y - wrist.y) / scale)
            row.append((lm.z - wrist.z) / scale)

        row.append(label)
        dataset.append(row)

print("\nWriting CSV file...")

header = [f"f{i}" for i in range(63)]
header.append("label")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)

print("\nCSV Created Successfully!")
print(f"Total Samples Saved: {len(dataset)}")
print(f"Skipped Images: {skipped_images}")
