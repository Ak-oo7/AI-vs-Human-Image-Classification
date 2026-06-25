"""
train.py
--------
Loads preprocessed feature-stack images, builds a CNN, trains it for 20 epochs,
and saves the trained model to models/ai_vs_real_classifier.h5

Usage:
    python src/train.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed_dataset")
CSV_PATH         = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_SAVE_PATH  = os.path.join(BASE_DIR, "models", "ai_vs_real_classifier.h5")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

IMAGE_SIZE  = (64, 64)
EPOCHS      = 20
BATCH_SIZE  = 32
RANDOM_SEED = 42


# ── Load data ─────────────────────────────────────────────────────────────────
def load_dataset(preprocessed_dir: str, csv_path: str):
    """
    Reads the label CSV and loads the matching preprocessed images.

    Returns:
        X : np.ndarray  shape (N, 64, 64, 3)  float32 in [0, 1]
        y : np.ndarray  shape (N,)             int {0=Real, 1=AI}
    """
    df          = pd.read_csv(csv_path)
    df["file_name"] = df["file_name"].apply(os.path.basename)
    labels_dict = dict(zip(df["file_name"], df["label"]))

    X, y = [], []

    for img_name in tqdm(os.listdir(preprocessed_dir), desc="Loading images"):
        if img_name not in labels_dict:
            continue

        img_path = os.path.join(preprocessed_dir, img_name)
        img      = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[WARN] Could not load: {img_path}")
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0   # normalise to [0, 1]

        X.append(img)
        y.append(labels_dict[img_name])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Ensure channel dim exists
    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)

    return X, y


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(input_shape=(64, 64, 3)) -> Sequential:
    """
    Custom CNN:
        3 × Conv2D (32 / 64 / 128 filters) + MaxPooling
        Dense 128 + Dropout(0.5) + Sigmoid output
    """
    model = Sequential([
        Conv2D(32,  (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64,  (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1,   activation="sigmoid"),      # binary: AI=1, Real=0
    ])
    return model


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Loading dataset ──────────────────────────────────")
    X, y = load_dataset(PREPROCESSED_DIR, CSV_PATH)
    print(f"Dataset shape : X={X.shape}, y={y.shape}")
    print(f"Class balance : Real={np.sum(y==0)}, AI={np.sum(y==1)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train size : {len(X_train)} | Val size : {len(X_val)}")

    print("\n── Building model ───────────────────────────────────")
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], X.shape[-1]))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n── Training ─────────────────────────────────────────")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

    # Final metrics
    val_acc  = max(history.history["val_accuracy"])
    val_loss = min(history.history["val_loss"])
    print(f"\nBest val accuracy : {val_acc:.4f}")
    print(f"Best val loss     : {val_loss:.4f}")
