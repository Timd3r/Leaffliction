#!/usr/bin/env python3
from Transformation import Transformation
import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import zipfile

# Ensure project root is always in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Embedded feature extractor to bypass _path_setup dependency


def plots_to_features(plots, gaussian_mask):
    """Convert color_analysis plots list into numerical features for ML prediction."""
    features = []
    for group, channel_idx, name, _ in plots:
        hist = cv2.calcHist([group], [channel_idx],
                            gaussian_mask, [256], [0, 256])
        hist = hist.flatten()
        hist_sum = hist.sum()
        if hist_sum == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        hist_norm = hist / hist_sum
        bins = np.arange(256, dtype=float)
        mean = np.sum(bins * hist_norm)
        std = np.sqrt(np.sum(hist_norm * (bins - mean) ** 2))
        peak_pos = np.argmax(hist_norm)
        peak_val = hist_norm[peak_pos]
        hist_nonzero = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))
        features.extend([mean, std, peak_pos, peak_val, entropy])
    return features


def main():
    """ Main function to load model, process image, and predict disease."""
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    # Clean path from accidental escape characters
    image_path = image_path.replace("\\", "").strip()
    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        print(f" Image not found: {image_path}")
        sys.exit(1)

    # Locate the learnings ZIP
    zip_candidates = [
        "./output/Apple_learnings.zip",
        "./Apple_learnings.zip",
        "./output/Grape_learnings.zip",
        "./Grape_learnings.zip"
    ]
    zip_path = next((z for z in zip_candidates if os.path.exists(z)), None)

    if zip_path is None:
        print(" Learnings ZIP not found. Run train.py first.")
        sys.exit(1)

    print(f"📦 Loading model from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        config = json.loads(zf.read("config.json").decode())
        with zf.open("model.joblib") as mf:
            clf = joblib.load(mf)

    feature_names = config["feature_names"]

    print("🔍 Processing image...")
    t = Transformation(dest_dir="./temp_plots")
    t.load_image(image_path)
    t.gaussian_blur()

    if t.gaussian_mask is None or not np.any(t.gaussian_mask > 128):
        print("No valid leaf detected in image.")
        sys.exit(1)

    # Recreate plots list exactly as in color_analysis()
    hsv = cv2.cvtColor(t.img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(t.img_bgr, cv2.COLOR_BGR2LAB)
    plots = [
        (t.img_bgr, 2, 'blue', '#0000FF'),
        (lab, 2, 'blue-yellow', '#FFFF00'),
        (t.img_bgr, 1, 'green', '#008000'),
        (lab, 1, 'green-magenta', '#FF00FF'),
        (hsv, 0, 'hue', '#8A2BE2'),
        (lab, 0, 'lightness', '#696969'),
        (t.img_bgr, 0, 'red', '#FF0000'),
        (hsv, 1, 'saturation', '#00FFFF'),
        (hsv, 2, 'value', '#FFA500')
    ]

    features = plots_to_features(plots, t.gaussian_mask)

    # Align features exactly with training using pandas DataFrame
    if len(features) != len(feature_names):
        print(
            f" Feature mismatch! Expected {len(feature_names)}, got {len(features)}. Padding/trimming...")
        features = features[:len(feature_names)]
        features += [0.0] * max(0, len(feature_names) - len(features))

    # DataFrame ensures exact column order & names, eliminating sklearn warnings
    X_pred = pd.DataFrame([features], columns=feature_names)

    # PREDICTION & DISPLAY

    # 1. Get prediction (returns string label directly, e.g., 'scab')
    raw_pred = clf.predict(X_pred)[0]
    disease = str(raw_pred)

    # 2. Get confidence
    proba = clf.predict_proba(X_pred)[0]
    confidence = float(np.max(proba))

    print(f"\nPredicted Disease: {disease}")
    print(f"Confidence: {confidence*100:.1f}%")

    # 3. Display Original + Transformed side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(t.img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    transformed = t.background_removed if t.background_removed is not None else t.img_bgr
    axes[1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Transformed (Prediction: {disease})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
