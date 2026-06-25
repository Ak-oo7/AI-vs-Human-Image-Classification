# AI vs Human Generated Image Classification
### CNN with Wavelet (DWT) · DCT · Gabor Feature Extraction

> Binary classifier that detects whether an image was created by an AI generator or a human photographer.

**Team:** Abhinav I · Fawas P · Jifsha Jabir · Niharika Ranjith · Sreelakshmi M M · Usmanul Faris

---

## Project Overview

With the rapid growth of AI image generators (Stable Diffusion, Midjourney, DALL-E, etc.), distinguishing AI-generated images from real photographs is becoming increasingly important for media verification, content moderation, and digital forensics.

This project builds an end-to-end pipeline that:
1. Applies **three complementary signal-processing filters** to each image — DWT noise residuals, DCT frequency artifacts, and Gabor texture responses — and stacks them into a 3-channel feature image.
2. Trains a **custom CNN** on those feature images for binary classification (`0 = Real`, `1 = AI Generated`).
3. Achieves high validation accuracy and outputs confidence scores on new images.

---

## Repository Structure

```
AI-vs-Human-Image-Classification/
│
├── src/
│   ├── preprocess.py     # DWT + DCT + Gabor feature extraction & saving
│   ├── train.py          # CNN model definition, training, saving
│   ├── predict.py        # Single-image & batch inference with confidence scores
│   └── visualize.py      # EDA — raw image comparison + preprocessed samples
│
├── notebooks/
│   └── AI_vs_Human_Classification.ipynb   # Full walkthrough notebook
│
├── data/
│   └── README.txt        # Instructions for downloading the dataset
│
├── models/
│   └── README.txt        # Model weights go here after training
│
├── outputs/              # Saved plots, submission.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### Feature Engineering (Preprocessing)

Each grayscale image is transformed into a 3-channel **64 × 64 feature stack**:

| Channel | Transform | What it captures |
|---------|-----------|-----------------|
| 1 | Haar DWT — HH sub-band (`pywt`) | High-frequency noise residuals — AI generators leave distinct noise signatures |
| 2 | Discrete Cosine Transform (`cv2.dct`) | Frequency-domain compression artifacts — common in AI outputs |
| 3 | Gabor filter response (`cv2.getGaborKernel`) | Oriented texture patterns — AI images often have unnaturally smooth textures |

All three channels are normalised to [0, 255] and saved as uint8 PNG files.

### CNN Architecture

```
Input  64×64×3
  Conv2D(32,  3×3, relu) → MaxPool(2×2)
  Conv2D(64,  3×3, relu) → MaxPool(2×2)
  Conv2D(128, 3×3, relu) → MaxPool(2×2)
  Flatten
  Dense(128, relu) → Dropout(0.5)
  Dense(1, sigmoid)         ← binary output
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Epochs | 20 |
| Batch size | 32 |
| Train/Val split | 80 / 20 |

---

## Dataset

Source: [AI vs Human Generated Dataset — Kaggle](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)

Labels: `0 = Real (human-made)`, `1 = AI Generated`

Download via `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
```

Place `train.csv`, `test.csv`, `train_data/`, and `test_data_v2/` inside the `data/` directory.

---

## Installation

```bash
git clone https://github.com/<your-username>/AI-vs-Human-Image-Classification.git
cd AI-vs-Human-Image-Classification

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Step 1 — Preprocess training images
```bash
python src/preprocess.py
```
Generates `data/preprocessed_dataset/` (DWT + DCT + Gabor feature stacks).

### Step 2 — Train the CNN
```bash
python src/train.py
```
Saves model to `models/ai_vs_real_classifier.h5`.

### Step 3 — Predict on a single image
```bash
python src/predict.py --image path/to/your_image.jpg
```
Output example:
```
Prediction : AI Generated   (Confidence: 0.97)
```

### Step 4 — Batch predict on entire test folder
```bash
python src/predict.py --test_dir data/test_data_v2
```
Saves `outputs/submission.csv`.

### Or run everything in the notebook
```bash
jupyter lab notebooks/AI_vs_Human_Classification.ipynb
```

---

## Results

The CNN trained on DWT + DCT + Gabor feature stacks achieves:

- **High validation accuracy** across 20 epochs
- **Clear confidence separation** between AI and Real images  
  (e.g. AI: 1.00 confidence · Real: 0.73 confidence · AI composite: 0.90)

Sample output predictions:

| Image | Prediction | Confidence |
|-------|-----------|------------|
| Man with dog (AI) | AI Generated | 1.00 |
| Two people selfie (Real) | Real | 0.73 |
| Red panda composite (AI) | AI Generated | 0.90 |

---

## Key Design Choices

**Why hybrid preprocessing instead of raw pixels?**  
Most existing models operate purely on pixel values, which limits their ability to detect subtle generation artifacts. By explicitly encoding noise patterns (DWT), frequency irregularities (DCT), and texture structure (Gabor), the classifier receives features that are specifically informative for AI-generation detection, reducing the burden on the CNN to discover these representations from scratch.

**Why a custom CNN instead of a pretrained backbone?**  
The input is not a natural RGB image but an engineered 3-channel feature representation. Transfer-learning backbones trained on ImageNet are optimised for natural image statistics and would require significant domain adaptation. A purpose-built CNN trains faster and performs well on this specific feature space.

---

## Future Directions

- Extend to **video-based deepfake detection**
- Integrate **transformer-based models** (ViT, Swin) for better generalisation
- Replace hand-crafted filters with **learnable frequency layers**
- Expand to **cross-domain AI generators** (DALL-E, Midjourney, Stable Diffusion XL)
- **Deploy** as a web-based detection API

---

## License

MIT License — free to use, modify, and distribute with attribution.
