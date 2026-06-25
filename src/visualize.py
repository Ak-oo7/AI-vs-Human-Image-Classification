"""
visualize.py
------------
Exploratory Data Analysis:
  • Shows sample Real vs AI images side-by-side with Canny edges.
  • Displays a sample of preprocessed feature-stack images.

Usage:
    python src/visualize.py
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH         = os.path.join(BASE_DIR, "data", "train.csv")
DATASET_PATH     = os.path.join(BASE_DIR, "data", "train_data")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed_dataset")
TEST_CSV_PATH    = os.path.join(BASE_DIR, "data", "test.csv")
TEST_DATA_PATH   = os.path.join(BASE_DIR, "data", "test_data_v2")


# ── 1. Raw image comparison: AI vs Real with edge maps ───────────────────────
def visualize_image_differences(ai_images: list, real_images: list,
                                  dataset_path: str, num_images: int = 5) -> None:
    """
    For each pair (AI, Real) shows:
      col 0 – raw AI image
      col 1 – AI Canny edges
      col 2 – raw Real image
      col 3 – Real Canny edges
    """
    num_images = min(len(ai_images), len(real_images), num_images)
    fig, axes  = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))

    for i in range(num_images):
        for j, (label, img_list) in enumerate(
            zip(["AI Generated", "Real"], [ai_images, real_images])
        ):
            img_path = os.path.join(dataset_path, img_list[i])
            img      = cv2.imread(img_path)

            if img is None:
                print(f"[WARN] Cannot load: {img_path}")
                continue

            img_gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            col_offset = j * 2

            # Raw image
            axes[i, col_offset].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i, col_offset].set_title(f"{label} Image {i+1}")
            axes[i, col_offset].axis("off")

            # Canny edges
            edges = cv2.Canny(img_gray, 100, 200)
            axes[i, col_offset + 1].imshow(edges, cmap="gray")
            axes[i, col_offset + 1].set_title(f"{label} Canny Edges")
            axes[i, col_offset + 1].axis("off")

    plt.suptitle("AI Generated vs Real Images — with Canny Edge Maps", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "outputs", "eda_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → outputs/eda_comparison.png")


# ── 2. Preprocessed feature-stack samples ─────────────────────────────────────
def show_preprocessed_samples(preprocessed_dir: str, n: int = 5) -> None:
    """Display n random preprocessed 3-channel feature images as grayscale."""
    image_files = [
        f for f in os.listdir(preprocessed_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        print("[INFO] No preprocessed images found. Run preprocess.py first.")
        return

    samples = random.sample(image_files, min(n, len(image_files)))
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))

    for ax, img_name in zip(axes, samples):
        img_path = os.path.join(preprocessed_dir, img_name)
        img      = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[WARN] Cannot load: {img_path}")
            continue

        img_gray = np.mean(img, axis=-1).astype(np.uint8)
        ax.imshow(img_gray, cmap="gray")
        ax.set_title(img_name[:20], fontsize=8)
        ax.axis("off")

    plt.suptitle("Preprocessed Feature-Stack Samples (DWT + DCT + Gabor)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "outputs", "preprocessed_samples.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → outputs/preprocessed_samples.png")


# ── 3. Show test images ───────────────────────────────────────────────────────
def show_test_images(test_df: pd.DataFrame, test_data_path: str,
                      num_images: int = 5) -> None:
    """Display the first num_images from the test set."""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        img_path = os.path.join(test_data_path, os.path.basename(test_df.iloc[i]["id"]))
        img      = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Cannot load: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Test Image {i+1}")

    plt.suptitle("Sample Test Images", fontsize=12)
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    # --- Load train CSV and split by label
    if os.path.exists(CSV_PATH):
        df         = pd.read_csv(CSV_PATH)
        real_images = df[df["label"] == 0]["file_name"].tolist()
        ai_images   = df[df["label"] == 1]["file_name"].tolist()

        selected_ai   = np.random.choice(ai_images,   5, replace=False).tolist()
        selected_real = np.random.choice(real_images, 5, replace=False).tolist()

        print("── Raw image comparison (AI vs Real) ───────────────")
        visualize_image_differences(selected_ai, selected_real, DATASET_PATH)
    else:
        print(f"[INFO] Train CSV not found at {CSV_PATH}. Skipping raw comparison.")

    # --- Preprocessed samples
    if os.path.exists(PREPROCESSED_DIR):
        print("\n── Preprocessed feature-stack samples ─────────────")
        show_preprocessed_samples(PREPROCESSED_DIR)
    else:
        print(f"[INFO] Preprocessed dir not found. Run preprocess.py first.")
