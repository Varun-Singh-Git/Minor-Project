"""
feature_extraction.py
=====================
Extracts a 10-dimensional feature vector from each preprocessed leaf image.

Feature vector composition:
  [0] contrast      — GLCM: intensity variation between adjacent pixels
  [1] energy        — GLCM: uniformity / textural smoothness
  [2] homogeneity   — GLCM: closeness of distribution to diagonal
  [3] correlation   — GLCM: linear dependency between pixel pairs
  [4] red_mean      — Mean of red channel
  [5] green_mean    — Mean of green channel
  [6] blue_mean     — Mean of blue channel
  [7] red_std       — Std deviation of red channel
  [8] green_std     — Std deviation of green channel
  [9] blue_std      — Std deviation of blue channel

Usage:
    python src/feature_extraction.py
    -- Processes all images, builds feature matrix, saves as data/features.csv
"""

import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from preprocess import preprocess_image, preprocess_dataset, CLASS_NAMES, DATA_DIR

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
FEATURES_CSV   = "data/features.csv"
GLCM_DISTANCES = [1]          # pixel distance for GLCM computation
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]   # 0°, 45°, 90°, 135°


# ──────────────────────────────────────────────
# GLCM Texture Features
# ──────────────────────────────────────────────
def extract_glcm_features(img: np.ndarray) -> dict:
    """
    Compute four GLCM texture features from a preprocessed image.

    GLCM (Grey-Level Co-occurrence Matrix) captures spatial relationships
    between pixel intensities. Features are averaged across 4 angles to
    achieve rotation invariance.

    GLCM Features:
        Contrast    = Σ (i-j)² × P(i,j)
        Energy      = Σ P(i,j)²
        Homogeneity = Σ P(i,j) / (1 + |i-j|)
        Correlation = Σ [(i-μᵢ)(j-μⱼ)P(i,j)] / (σᵢσⱼ)

    Args:
        img: Preprocessed float32 image array (H, W, 3).

    Returns:
        Dictionary with keys: contrast, energy, homogeneity, correlation.
    """
    # Convert to grayscale for GLCM
    gray = np.mean(img, axis=2)           # simple average of RGB channels

    # Convert to uint8 (GLCM requires integer grey levels)
    gray_uint8 = (gray * 255).astype(np.uint8)

    # Compute GLCM matrix
    # Shape: (levels, levels, len(distances), len(angles))
    glcm = graycomatrix(
        gray_uint8,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract properties — mean over all distances and angles
    contrast    = graycoprops(glcm, "contrast").mean()
    energy      = graycoprops(glcm, "energy").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    correlation = graycoprops(glcm, "correlation").mean()

    return {
        "contrast":    float(contrast),
        "energy":      float(energy),
        "homogeneity": float(homogeneity),
        "correlation": float(correlation),
    }


# ──────────────────────────────────────────────
# Colour Features (RGB channel statistics)
# ──────────────────────────────────────────────
def extract_color_features(img: np.ndarray) -> dict:
    """
    Compute mean and standard deviation of each RGB channel.

    These 6 values capture overall colour distribution, which differs
    significantly between disease types (e.g., yellowing in Leaf Spot
    shows elevated red_mean, reduced green_mean).

    Args:
        img: Preprocessed float32 image array (H, W, 3).

    Returns:
        Dictionary with keys: red_mean, green_mean, blue_mean,
                               red_std, green_std, blue_std.
    """
    # Ignore black background pixels (zeroed out during masking)
    mask = img.sum(axis=2) > 0.0   # True where at least one channel > 0

    if mask.sum() == 0:
        # Edge case: fully masked image — return zeros
        return {
            "red_mean": 0.0, "green_mean": 0.0, "blue_mean": 0.0,
            "red_std":  0.0, "green_std":  0.0, "blue_std":  0.0,
        }

    r = img[:, :, 0][mask]
    g = img[:, :, 1][mask]
    b = img[:, :, 2][mask]

    return {
        "red_mean":   float(r.mean()),
        "green_mean": float(g.mean()),
        "blue_mean":  float(b.mean()),
        "red_std":    float(r.std()),
        "green_std":  float(g.std()),
        "blue_std":   float(b.std()),
    }


# ──────────────────────────────────────────────
# Combined 10-D feature vector
# ──────────────────────────────────────────────
def build_feature_vector(img: np.ndarray) -> dict:
    """
    Build the complete 10-dimensional feature vector for one image.

    Combines GLCM texture features (4) + colour statistics (6).

    Args:
        img: Preprocessed float32 image array.

    Returns:
        Ordered dictionary with all 10 features.
    """
    glcm_feats  = extract_glcm_features(img)
    color_feats = extract_color_features(img)
    return {**glcm_feats, **color_feats}   # merge both dicts


# ──────────────────────────────────────────────
# Single image prediction helper
# ──────────────────────────────────────────────
def extract_features_from_path(image_path: str) -> np.ndarray:
    """
    Full pipeline: load raw image → preprocess → extract features.

    Used by predict.py and app.py for single-image inference.

    Args:
        image_path: Path to the raw leaf image.

    Returns:
        NumPy array of shape (1, 10) — ready for model.predict()
    """
    img = preprocess_image(image_path)
    feature_dict = build_feature_vector(img)
    feature_array = np.array(list(feature_dict.values())).reshape(1, -1)
    return feature_array


# ──────────────────────────────────────────────
# Build full dataset feature matrix
# ──────────────────────────────────────────────
def build_dataset_features(data_dir: str = DATA_DIR,
                            output_csv: str = FEATURES_CSV) -> pd.DataFrame:
    """
    Process all images in the dataset and save the feature matrix to CSV.

    Args:
        data_dir:   Root folder with class subfolders.
        output_csv: Output path for the features CSV.

    Returns:
        DataFrame with shape (n_images, 11) — 10 features + label column.
    """
    images, labels = preprocess_dataset(data_dir)

    print(f"\nExtracting features from {len(images)} images...")
    rows = []

    for i, (img, label) in enumerate(zip(images, labels)):
        if i % 200 == 0:
            print(f"  [{i}/{len(images)}] processing...")

        feature_dict = build_feature_vector(img)
        feature_dict["label"] = label
        rows.append(feature_dict)

    df = pd.DataFrame(rows)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Features saved → {output_csv}")
    print(f"     Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Crop Disease Detection — Feature Extraction")
    print("=" * 55)

    df = build_dataset_features()

    print("\nSample rows:")
    print(df.head())

    print("\nFeature statistics:")
    print(df.describe().round(4))
