"""
GestureOS — train.py
─────────────────────
Standalone training script. Run this to (re)train the model
from collected gesture_data/*.csv files.

Usage:
    python train.py
    python train.py --data_dir gesture_data --model gesture_model.pkl

The same training logic is also built into server.py (triggered via
the frontend "Train Model" button), so you can use either approach.
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(data_dir: Path):
    X, y, label_to_idx = [], [], {}

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERR] No CSV files in {data_dir}/")
        print("      Record gestures first using the frontend + server.py")
        sys.exit(1)

    print(f"\n{'─'*50}")
    print(f"  Loading dataset from: {data_dir}")
    print(f"{'─'*50}")

    for csv_path in csv_files:
        gesture_name = csv_path.stem
        idx = len(label_to_idx)
        label_to_idx[gesture_name] = idx
        count = 0

        with open(csv_path, "r") as f:
            for line in f:
                vals = [float(v) for v in line.strip().split(",") if v]
                if len(vals) == 63:
                    X.append(vals)
                    y.append(idx)
                    count += 1

        print(f"  [{idx}] {gesture_name:<25} {count:>5} frames")

    print(f"{'─'*50}")
    print(f"  Total: {len(X)} samples, {len(label_to_idx)} classes")
    print(f"{'─'*50}\n")

    return np.array(X), np.array(y), label_to_idx


def train(data_dir: Path, model_path: Path, label_path: Path):
    from sklearn.neural_network   import MLPClassifier
    from sklearn.model_selection  import train_test_split, cross_val_score
    from sklearn.preprocessing    import StandardScaler
    from sklearn.pipeline         import Pipeline
    from sklearn.metrics          import classification_report

    X, y, label_to_idx = load_dataset(data_dir)

    if len(set(y)) < 2:
        print("[ERR] Need at least 2 gesture classes to train.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=False,
        )),
    ])

    print("  Training MLPClassifier...")
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"\n  Test accuracy : {acc:.2%}")

    # Detailed report
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
    print("\n" + classification_report(y_test, clf.predict(X_test), target_names=target_names))

    # Save model + labels
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(label_path, "wb") as f:
        pickle.dump(idx_to_label, f)

    print(f"  Model saved  → {model_path}")
    print(f"  Labels saved → {label_path}")
    print(f"\n  Done! Start server.py and open index.html to use inference.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GestureOS gesture classifier")
    parser.add_argument("--data_dir", default="gesture_data",  help="Folder with gesture CSVs")
    parser.add_argument("--model",    default="gesture_model.pkl", help="Output model file")
    parser.add_argument("--labels",   default="gesture_labels.pkl", help="Output labels file")
    args = parser.parse_args()

    base = Path(__file__).parent
    train(
        data_dir   = base / args.data_dir,
        model_path = base / args.model,
        label_path = base / args.labels,
    )
