import numpy as np
import cv2
import os
import csv
import time
from tqdm import tqdm
import _path_setup
import Transformation


def plots_to_features(plots, gaussian_mask):
    """
    Convert color_analysis() plots list into numerical features for ML training.
    Always returns exactly 45 features (9 channels × 5 statistics).
    """
    features = []

    for group, channel_idx, name, _ in plots:
        # Compute histogram using ONLY leaf pixels
        hist = cv2.calcHist([group], [channel_idx],
                            gaussian_mask, [256], [0, 256])
        hist = hist.flatten()

        hist_sum = hist.sum()
        if hist_sum == 0:
            # Return zeros instead of skipping to keep vector length consistent
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        hist_norm = hist / hist_sum
        bins = np.arange(256, dtype=float)

        # Extract 5 key statistics per channel
        mean = np.sum(bins * hist_norm)
        std = np.sqrt(np.sum(hist_norm * (bins - mean) ** 2))
        peak_pos = np.argmax(hist_norm)
        peak_val = hist_norm[peak_pos]

        # Entropy (measure of distribution uniformity)
        hist_nonzero = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))

        features.extend([mean, std, peak_pos, peak_val, entropy])

    return features


def count_images_in_folder(folder_path):
    """Count total .jpg images in a folder recursively."""
    total = 0
    for root, _, files in os.walk(folder_path):
        total += sum(1 for f in files if f.lower().endswith(".jpg"))
    return total


def create_dataset_for_category(parent, split, output_csv):
    """
    Create a dataset CSV for a specific parent (Apple/Grape) and split (train/test).

    Args:
        parent: "Apple" or "Grape"
        split: "train" or "test"
        output_csv: Path to output CSV file
    """
    base_path = f"./data/{split}/{parent}"

    if not os.path.exists(base_path):
        print(f"⚠️ Warning: {base_path} not found. Skipping.")
        return 0, 0

    # Define expected channel names (matches color_analysis order)
    channel_names = ['blue', 'blue-yellow', 'green', 'green-magenta',
                     'hue', 'lightness', 'red', 'saturation', 'value']
    stats = ['mean', 'std', 'peak_pos', 'peak_val', 'entropy']

    # Dynamically build header to guarantee alignment
    feature_cols = [f"{ch}_{st}" for ch in channel_names for st in stats]
    header = ["image_path", "label"] + feature_cols

    # Count total images for progress bar
    total_images = count_images_in_folder(base_path)
    if total_images == 0:
        print(f"⚠️ No images found in {base_path}. Skipping.")
        return 0, 0

    processed_count = 0
    error_count = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Create progress bar
        with tqdm(total=total_images, desc=f"Processing {parent} {split}",
                  unit="img", ncols=100) as pbar:

            for category in os.listdir(base_path):
                category_path = os.path.join(base_path, category)
                if not os.path.isdir(category_path):
                    continue

                for root, _, files in os.walk(category_path):
                    for file in files:
                        if not file.lower().endswith(".jpg"):
                            continue

                        image_path = os.path.join(root, file)

                        try:
                            # Initialize with dest_dir to suppress plt.show() during batch run
                            t = Transformation.Transformation(
                                dest_dir="./temp_plots")
                            t.load_image(image_path)
                            t.gaussian_blur()  # Sets self.gaussian_mask

                            if t.gaussian_mask is None or not np.any(t.gaussian_mask > 128):
                                raise ValueError("No valid leaf mask detected")

                            plots = t.color_analysis()  # Returns the plots list
                            features = plots_to_features(
                                plots, t.gaussian_mask)

                            # Safety check: ensure exactly 45 features
                            if len(features) != 45:
                                features = features[:45]
                                features += [0.0] * (45 - len(features))

                            # Write row: [path, label, feat1, feat2, ..., feat45]
                            writer.writerow([image_path, category] + features)
                            processed_count += 1

                        except Exception as e:
                            error_count += 1
                            # Update progress bar description with error count
                            pbar.set_postfix(
                                {'errors': error_count}, refresh=False)

                        # Update progress bar
                        pbar.update(1)

    return processed_count, error_count


def main():
    """Main function to create all 4 datasets."""
    print("="*60)
    print("🚀 Starting Dataset Creation")
    print("="*60)

    start_time = time.time()

    # Define the 4 datasets to create
    datasets = [
        ("Apple", "train", "./train_apple.csv"),
        ("Apple", "test", "./test_apple.csv"),
        ("Grape", "train", "./train_grape.csv"),
        ("Grape", "test", "./test_grape.csv"),
    ]

    total_processed = 0
    total_errors = 0

    for parent, split, output_csv in datasets:
        print(f"\n{'='*60}")
        print(f"📊 Creating {parent.upper()} {split.upper()} dataset...")
        print(f"{'='*60}")

        processed, errors = create_dataset_for_category(
            parent, split, output_csv)

        total_processed += processed
        total_errors += errors

        print(f"\n✅ {parent} {split} complete:")
        print(f"   📈 Processed: {processed} images")
        print(f"   ❌ Errors: {errors} images")
        print(f"   💾 Saved to: {output_csv}")

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("🎉 All datasets created successfully!")
    print(f"{'='*60}")
    print(f"📊 Total processed: {total_processed} images")
    print(f"⚠️ Total errors: {total_errors} images")
    print(f"⏱️ Total time: {total_time/60:.2f} minutes")
    print(f"\n📁 Output files:")
    print(f"   - train_apple.csv")
    print(f"   - test_apple.csv")
    print(f"   - train_grape.csv")
    print(f"   - test_grape.csv")
    print("="*60)


if __name__ == "__main__":
    main()
