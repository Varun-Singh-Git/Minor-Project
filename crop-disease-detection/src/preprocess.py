"""
preprocess.py
=============
Image preprocessing pipeline for the Crop Disease Detection System.

Pipeline stages:
  1. Load image from disk
  2. Resize to 128x128 pixels
  3. Normalize pixel values to [0.0, 1.0]
  4. Apply green-channel mask to isolate leaf from background
  5. Apply Otsu's binarization for segmentation

Usage:
    python src/preprocess.py
    -- Processes all images in data/images/ and saves segmented arrays.
    -- Or import individual functions into other scripts.
"""

import os
import cv2
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
IMAGE_SIZE = (128, 128)          # target width x height
DATA_DIR   = "data/images"       # root folder containing class subfolders
CLASS_NAMES = [
    "Healthy",
    "Leaf_Spot",
    "Leaf_Curl",
    "Slug_Damage",
    "Fruit_Disease"
]


# ──────────────────────────────────────────────
# Step 1 — Load image
# ──────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as an RGB NumPy array.

    Args:
        path: Full path to the image file.

    Returns:
        NumPy array of shape (H, W, 3) in RGB colour order.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR
    return img_rgb


# ──────────────────────────────────────────────
# Step 2 — Resize image
# ──────────────────────────────────────────────
def resize_image(img: np.ndarray, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Resize the image to a uniform resolution.

    Args:
        img:  RGB NumPy array.
        size: (width, height) tuple, default 128x128.

    Returns:
        Resized NumPy array of shape (size[1], size[0], 3).
    """
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized


# ──────────────────────────────────────────────
# Step 3 — Normalize pixel values
# ──────────────────────────────────────────────
def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel intensity values from [0, 255] to [0.0, 1.0].

    This reduces the effect of lighting variation across images taken
    in different field conditions.

    Args:
        img: NumPy array with dtype uint8 (values 0–255).

    Returns:
        NumPy float32 array with values in [0.0, 1.0].
    """
    return img.astype(np.float32) / 255.0


# ──────────────────────────────────────────────
# Step 4 — Green-channel masking
# ──────────────────────────────────────────────
def green_channel_mask(img: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """
    Isolate leaf pixels using the green channel.

    The green channel is the dominant channel in healthy vegetation.
    Pixels where the green channel significantly exceeds the red and blue
    channels are retained; background soil/sky pixels are zeroed out.

    Args:
        img:       Normalized RGB float32 array.
        threshold: Minimum green dominance margin (default 0.15).

    Returns:
        Masked image array of same shape, background pixels set to 0.
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Pixel is "green-dominant" if green channel exceeds both R and B by threshold
    green_dominant = (g > r + threshold) | (g > b + threshold)

    mask = green_dominant.astype(np.float32)
    masked = img * mask[:, :, np.newaxis]   # apply mask to all 3 channels
    return masked


# ──────────────────────────────────────────────
# Step 5 — Segmentation using Otsu's binarization
# ──────────────────────────────────────────────
def segment_image(img: np.ndarray) -> np.ndarray:
    """
    Segment the leaf region using Otsu's automatic thresholding.

    Converts the masked image to grayscale, applies Otsu's binarization
    to produce a binary mask, then applies it back to the colour image
    so that only the leaf region (potential disease area) is retained.

    Args:
        img: Normalized, green-masked RGB float32 array.

    Returns:
        Segmented image — same shape as input, non-leaf pixels zeroed.
    """
    # Convert to uint8 for OpenCV threshold
    img_uint8 = (img * 255).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Otsu's thresholding — automatically finds optimal threshold
    _, binary_mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup to remove tiny noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Apply mask back to the normalised float image
    mask_float = clean_mask.astype(np.float32) / 255.0
    segmented = img * mask_float[:, :, np.newaxis]

    return segmented


# ──────────────────────────────────────────────
# Full pipeline — single image
# ──────────────────────────────────────────────
def preprocess_image(path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.

    Stages:  Load → Resize → Normalize → Green-mask → Segment

    Args:
        path: Path to the raw input image.

    Returns:
        Fully preprocessed float32 image array (128, 128, 3).
    """
    img = load_image(path)
    img = resize_image(img)
    img = normalize_image(img)
    img = green_channel_mask(img)
    img = segment_image(img)
    return img


# ──────────────────────────────────────────────
# Batch processing — all images in data/images/
# ──────────────────────────────────────────────
def preprocess_dataset(data_dir: str = DATA_DIR) -> tuple:
    """
    Run full preprocessing pipeline on all images in the dataset.

    Args:
        data_dir: Root folder containing one subfolder per class.

    Returns:
        (images, labels): Two lists — preprocessed arrays and string labels.
    """
    images = []
    labels = []
    skipped = 0

    for class_name in CLASS_NAMES:
        class_folder = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_folder):
            print(f"[WARNING] Folder not found: {class_folder}")
            continue

        files = [f for f in os.listdir(class_folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"Processing {len(files)} images in class '{class_name}' ...")

        for fname in files:
            fpath = os.path.join(class_folder, fname)
            try:
                img = preprocess_image(fpath)
                images.append(img)
                labels.append(class_name)
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
                skipped += 1

    print(f"\nTotal processed: {len(images)} | Skipped: {skipped}")
    return images, labels


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Crop Disease Detection — Preprocessing Pipeline")
    print("=" * 55)

    images, labels = preprocess_dataset()

    print(f"\nClass distribution:")
    from collections import Counter
    for cls, count in Counter(labels).items():
        print(f"  {cls:<20} {count} images")

    print("\n[OK] Preprocessing complete.")
    print("     Run src/feature_extraction.py next.")
