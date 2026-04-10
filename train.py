#!/usr/bin/env python3
"""
train.py - Learns disease characteristics and packages everything in a .zip
Usage: python3 train.py ./Apple/
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
import zipfile
import shutil


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train.py <directory>")
        sys.exit(1)

    base_dir = sys.argv[1].rstrip('/')
    print(f"Loading datasets for {base_dir}...")

    # Load your pre-split CSVs
    train_path = f"train_{base_dir.lower()}.csv"
    test_path = f"test_{base_dir.lower()}.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("train/test CSVs not found. Run create_dataset.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Clean labels (remove plant prefix like 'Apple_')
    train_df['label_clean'] = train_df['label'].str.replace(
        f'{os.path.basename(base_dir)}_', '', regex=False)
    test_df['label_clean'] = test_df['label'].str.replace(
        f'{os.path.basename(base_dir)}_', '', regex=False)

    # Separate features and labels
    feature_cols = [c for c in train_df.columns if c not in [
        'image_path', 'label', 'label_clean']]
    X_train, y_train = train_df[feature_cols], train_df['label_clean']
    X_test, y_test = test_df[feature_cols], test_df['label_clean']

    print(
        f"Training samples: {len(X_train)} | Validation samples: {len(X_test)}")
    if len(X_test) < 100:
        print("⚠️ Warning: Validation set has <100 images. Accuracy proof may not meet requirements.")

    # Train model
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Validate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    print(f"\nValidation Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # Prepare output directory
    os.makedirs("output", exist_ok=True)

    # Save model
    joblib.dump(clf, "output/model.joblib")

    # Save config
    config = {
        "feature_names": feature_cols,
        "label_mapping": dict(enumerate(clf.classes_)),
        "preprocessing": "color_histogram_stats (45 features)",
        "validation_accuracy": acc,
        "validation_size": len(X_test)
    }
    with open("output/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save validation report
    with open("output/validation_report.txt", "w") as f:
        f.write(f"Validation Accuracy: {acc*100:.2f}%\n")
        f.write(f"Validation Set Size: {len(X_test)} images\n")
        f.write(f"\nClassification Report:\n{report}")
        f.write(f"\nConfusion Matrix:\n{cm}")

    # Create ZIP as required
    zip_path = f"output/{os.path.basename(base_dir)}_learnings.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write("output/model.joblib", "model.joblib")
        zf.write("output/config.json", "config.json")
        zf.write("output/validation_report.txt", "validation_report.txt")

        # Add augmented training images
        aug_dir = os.path.join("./data/train", os.path.basename(base_dir))
        if os.path.exists(aug_dir):
            for root, _, files in os.walk(aug_dir):
                for file in files:
                    if file.endswith('.JPG'):
                        abs_path = os.path.join(root, file)
                        arcname = os.path.relpath(
                            abs_path, start="./data/train")
                        zf.write(abs_path, f"augmented_images/{arcname}")

    print(f"\nAll learnings packaged in: {zip_path}")
    print("Ready for evaluation!")


if __name__ == "__main__":
    main()
