"""
predict.py
----------
Runs inference on a single image or a full test directory.

Usage – single image:
    python src/predict.py --image path/to/image.jpg

Usage – full test folder (generates submission.csv):
    python src/predict.py --test_dir data/test_data_v2
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "ai_vs_real_classifier.h5")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "outputs", "submission.csv")

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

IMAGE_SIZE = (64, 64)


# ── Feature extractors (same as preprocess.py) ────────────────────────────────
def compute_wavelet_noise(img_gray: np.ndarray) -> np.ndarray:
    coeffs2 = pywt.dwt2(img_gray, "haar")
    _, (LH, HL, HH) = coeffs2
    return cv2.resize(np.abs(HH), (img_gray.shape[1], img_gray.shape[0]))


def compute_dct_artifacts(img_gray: np.ndarray) -> np.ndarray:
    dct = cv2.dct(np.float32(img_gray))
    return cv2.resize(
        np.log(np.abs(dct) + 1), (img_gray.shape[1], img_gray.shape[0])
    )


def compute_gabor(img_gray: np.ndarray) -> np.ndarray:
    gabor_kernel = cv2.getGaborKernel(
        (5, 5), 1.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F
    )
    return cv2.resize(
        cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel),
        (img_gray.shape[1], img_gray.shape[0]),
    )


def preprocess_image(image_path: str):
    """
    Load an image, apply the 3-filter pipeline, return
    (feature_stack float32 64×64×3,  original_gray uint8).
    Returns (None, None) if the image cannot be read.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Image not found: {image_path}")
        return None, None

    noise_residual = compute_wavelet_noise(img)
    dct_artifacts  = compute_dct_artifacts(img)
    gabor_response = compute_gabor(img)

    # Normalise each channel
    noise_residual = cv2.normalize(noise_residual, None, 0, 255, cv2.NORM_MINMAX)
    dct_artifacts  = cv2.normalize(dct_artifacts,  None, 0, 255, cv2.NORM_MINMAX)
    gabor_response = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX)

    feature_stack = np.dstack(
        [noise_residual, dct_artifacts, gabor_response]
    ).astype(np.uint8)

    feature_stack = cv2.resize(feature_stack, IMAGE_SIZE)
    feature_stack = feature_stack / 255.0   # float [0, 1]

    return feature_stack, img


# ── Single-image prediction ───────────────────────────────────────────────────
def predict_single(model, image_path: str, show_plot: bool = True):
    input_image, original_image = preprocess_image(image_path)

    if input_image is None:
        return

    input_tensor = np.expand_dims(input_image, axis=0)
    raw_pred     = model.predict(input_tensor, verbose=0)[0][0]

    label      = "AI Generated" if raw_pred > 0.5 else "Real"
    confidence = raw_pred if raw_pred > 0.5 else 1 - raw_pred

    print(f"Prediction : {label}   (Confidence: {confidence:.2f})")

    if show_plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(original_image, cmap="gray")
        plt.title(f"Prediction: {label}\nConfidence: {confidence:.2f}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return label, confidence


# ── Batch prediction on test folder ──────────────────────────────────────────
def predict_batch(model, test_dir: str, output_csv: str = OUTPUT_CSV_PATH):
    predictions = []

    image_files = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_name in tqdm(image_files, desc="Predicting"):
        img_path    = os.path.join(test_dir, img_name)
        input_image, _ = preprocess_image(img_path)

        if input_image is None:
            continue

        input_tensor = np.expand_dims(input_image, axis=0)
        raw_pred     = model.predict(input_tensor, verbose=0)[0][0]
        label        = "AI Generated" if raw_pred > 0.5 else "Real"

        predictions.append({"image_name": img_name, "predicted_label": label})

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to: {output_csv}")
    print(df["predicted_label"].value_counts())
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI vs Human Image Classifier — Inference"
    )
    parser.add_argument("--image",    type=str, help="Path to a single image")
    parser.add_argument("--test_dir", type=str, help="Path to test image folder")
    args = parser.parse_args()

    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    if args.image:
        predict_single(model, args.image)
    elif args.test_dir:
        predict_batch(model, args.test_dir)
    else:
        parser.print_help()
