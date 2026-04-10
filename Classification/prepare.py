import _path_setup  # Must be the very first import
import Augmentation
import Distribution
import os
import random
import math
import shutil


# 80/20 split of the data for training and testing
random.seed(42)


def make_dirs():
    """
    makes directories in . called data with Apple and Grape in
    it with the same subdirectories as the original dataset in
    also copies the images from the original dataset to the new data directory
    """
    base_path = "./leaves/images"
    data_path = "./data"
    os.makedirs(data_path, exist_ok=True)
    for parent in ["Apple", "Grape"]:
        parent_path = os.path.join(base_path, parent)
        for category in os.listdir(parent_path):
            category_path = os.path.join(parent_path, category)
            if os.path.isdir(category_path):
                new_category_path = os.path.join(data_path, parent, category)
                os.makedirs(new_category_path, exist_ok=True)
                for file in os.listdir(category_path):
                    if file.endswith(".JPG"):
                        src_file = os.path.join(category_path, file)
                        dst_file = os.path.join(new_category_path, file)
                        shutil.copy(src_file, dst_file)
    os.makedirs('./data/train', exist_ok=True)
    os.makedirs('./data/test', exist_ok=True)
    print("Directories created successfully.")


def balance_datasets(counts):
    """
    Balances dataset by augmenting underrepresented categories.
    - Uses the provided counts to determine how many augmentations are needed
    - Tracks used files to avoid duplicates across calls
    """
    used_tracker = {}

    max_count = max(counts.values()) if counts else 0

    for category, current_count in counts.items():
        gap = max_count - current_count
        if gap <= 0:
            continue

        num_to_augment = math.ceil(gap / 6)

        parent = "Apple" if "Apple" in category else "Grape"
        dir_path = os.path.join("./data/train", parent, category)

        if not os.path.isdir(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        # 1. Grab ONLY files that actually exist (ignores numbering completely)
        all_files = [
            f for f in os.listdir(dir_path)
            if f.lower().endswith(('.jpg', '.jpeg'))
        ]

        if not all_files:
            continue

        # 2. Remove files already used in previous calls
        already_used = used_tracker.get(category, set())
        available = [f for f in all_files if f not in already_used]

        # 3. Fallback: if we exhaust unique images, allow reuse with warning
        if len(available) < num_to_augment:
            print(
                f"'{category}': Only {len(available)} unused images. Reusing pool.")
            available = all_files

        # 4. Safe unique sampling
        num_to_pick = min(num_to_augment, len(available))
        chosen_files = random.sample(available, num_to_pick)

        # 5. Update tracker
        used_tracker.setdefault(category, set()).update(chosen_files)

        # 6. Augment
        for filename in chosen_files:
            image_path = os.path.join(dir_path, filename)
            os.makedirs(
                f'./data/train/{parent}/{category}/augmented', exist_ok=True)
            Augmentation.augment_image(
                image_path, save_location='./data/train/' + parent + '/' + category + '/augmented')
        print(f"'{category}': Augmented {num_to_pick} images to fill gap of {gap}.")


def create_train_test_split():
    """
    Splits the data in ./data into train and test folders with an random 80/20 split
    the test folder has the sub directories Apple and Grape with the same subdirectories as the original dataset in
     and the train folder doesn't and the file has a random name (image_(random number).JPG)
    """
    for parent in ["Apple", "Grape"]:
        parent_path = os.path.join("./data", parent)
        for category in os.listdir(parent_path):
            category_path = os.path.join(parent_path, category)
            if os.path.isdir(category_path):
                files = [f for f in os.listdir(
                    category_path) if f.endswith(".JPG")]
                random.shuffle(files)
                split_index = int(0.8 * len(files))
                train_files = files[:split_index]
                test_files = files[split_index:]

                for file in train_files:
                    src_file = os.path.join(category_path, file)
                    dst_file = os.path.join(
                        "./data/train", parent, category, file)
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy(src_file, dst_file)

                for file in test_files:
                    src_file = os.path.join(category_path, file)
                    file_name = f"image_{random.randint(1, 100000)}_{category}.JPG"
                    dst_file = os.path.join(
                        "./data/test", parent, file_name)
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy(src_file, dst_file)

    print("Train-test split created successfully.")
    # print train apple and grape counts and test apple and grape counts
    train_apple_count = sum(len(files)
                            for _, _, files in os.walk("./data/train/Apple"))
    train_grape_count = sum(len(files)
                            for _, _, files in os.walk("./data/train/Grape"))
    test_apple_count = sum(len(files)
                           for _, _, files in os.walk("./data/test/Apple"))
    test_grape_count = sum(len(files)
                           for _, _, files in os.walk("./data/test/Grape"))
    print(f"Train Apple Count: {train_apple_count}")
    print(f"Train Grape Count: {train_grape_count}")
    print(f"Test Apple Count: {test_apple_count}")
    print(f"Test Grape Count: {test_grape_count}")


def main():
    make_dirs()
    create_train_test_split()
    apple_counts = Distribution.run_distribution_analysis(
        './data/train/Apple')
    grape_counts = Distribution.run_distribution_analysis(
        './data/train/Grape')
    print("Apple Counts:", apple_counts)
    print("Grape Counts:", grape_counts)
    balance_datasets(apple_counts)
    balance_datasets(grape_counts)


if __name__ == "__main__":
    main()
