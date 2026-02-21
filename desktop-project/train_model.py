import pandas as pd
import numpy as np
import glob
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

csv_files = glob.glob("data/*.csv")
if not csv_files:
    print("No training data found")
    exit()

df = pd.concat([pd.read_csv(f) for f in csv_files])
df = df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

def add_orientation(X):
    wrist = X[:, 0:3]
    thumb = X[:, 12:15]
    index = X[:, 24:27]
    thumb_orientation = (thumb - wrist) * 2.0
    index_orientation = index - wrist
    return np.hstack([X, thumb_orientation, index_orientation])

X = add_orientation(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
joblib.dump(le, "label_encoder.pkl")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(69,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train,
          epochs=50,
          batch_size=64,
          validation_data=(X_test, y_test))

model.save("gesture_model.keras")

loss, acc = model.evaluate(X_test, y_test)
print(f"Final Accuracy: {acc}")
