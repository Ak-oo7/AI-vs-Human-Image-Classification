"""
preprocess.py
-------------
Applies Wavelet (DWT), DCT, and Gabor feature extraction to each training image
and saves the resulting 3-channel feature-stack to preprocessed_dataset/.

Usage:
    python src/preprocess.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH     = os.path.join(BASE_DIR, "data", "train.csv")
DATASET_PATH = os.path.join(BASE_DIR, "data", "train_data")
OUTPUT_DIR   = os.path.join(BASE_DIR, "data", "preprocessed_dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Feature extractors ───────────────────────────────────────────────────────
def compute_wavelet_noise(img_gray: np.ndarray) -> np.ndarray:
    """Extract high-frequency noise residuals using Haar DWT."""
    coeffs2 = pywt.dwt2(img_gray, "haar")
    _, (LH, HL, HH) = coeffs2
    return np.abs(HH)


def compute_dct_artifacts(img_gray: np.ndarray) -> np.ndarray:
    """Extract frequency-domain artifacts using DCT (log-scaled)."""
    dct = cv2.dct(np.float32(img_gray))
    return np.log(np.abs(dct) + 1)


def compute_gabor(img_gray: np.ndarray) -> np.ndarray:
    """Extract texture patterns using a Gabor filter."""
    gabor_kernel = cv2.getGaborKernel(
        (5, 5), 1.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F
    )
    return cv2.filter2D(img_gray, cv2.CV_8UC3, gabor_kernel)


# ── Core pipeline ─────────────────────────────────────────────────────────────
def process_and_save_images(image_list: list, dataset_path: str,
                             output_dir: str, img_size: int = 64) -> None:
    """
    For each image in image_list:
      1. Read as grayscale
      2. Compute DWT noise, DCT artifacts, Gabor response
      3. Resize all three to img_size × img_size
      4. Normalize to [0, 255]
      5. Stack as a 3-channel uint8 image and save
    """
    for img_name in tqdm(image_list, desc="Preprocessing images", unit="img"):
        img_path = os.path.join(dataset_path, os.path.basename(img_name))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARN] Could not load: {img_path}")
            continue

        noise_residual  = compute_wavelet_noise(img)
        dct_artifacts   = compute_dct_artifacts(img)
        gabor_response  = compute_gabor(img)

        # Resize to target size
        noise_residual  = cv2.resize(noise_residual,  (img_size, img_size))
        dct_artifacts   = cv2.resize(dct_artifacts,   (img_size, img_size))
        gabor_response  = cv2.resize(gabor_response,  (img_size, img_size))

        # Normalize each channel to [0, 255]
        noise_residual  = cv2.normalize(noise_residual,  None, 0, 255, cv2.NORM_MINMAX)
        dct_artifacts   = cv2.normalize(dct_artifacts,   None, 0, 255, cv2.NORM_MINMAX)
        gabor_response  = cv2.normalize(gabor_response,  None, 0, 255, cv2.NORM_MINMAX)

        # Stack → 3-channel feature image
        feature_stack = np.dstack(
            [noise_residual, dct_artifacts, gabor_response]
        ).astype(np.uint8)

        save_path = os.path.join(output_dir, os.path.basename(img_name))
        success   = cv2.imwrite(save_path, feature_stack)

        if not success:
            print(f"[WARN] Failed to save: {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df         = pd.read_csv(CSV_PATH)
    image_list = df["file_name"].tolist()

    print(f"Total images to preprocess : {len(image_list)}")
    print(f"Input  directory           : {DATASET_PATH}")
    print(f"Output directory           : {OUTPUT_DIR}")

    process_and_save_images(image_list, DATASET_PATH, OUTPUT_DIR)
    print(f"\n✅ Preprocessing complete! Saved to: {OUTPUT_DIR}")
