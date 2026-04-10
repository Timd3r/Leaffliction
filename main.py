import Distribution
import Augmentation


import random
import math


def balance_datasets(counts):
    max_count = max(counts.values())

    for category, current_count in counts.items():
        gap = max_count - current_count
        if gap <= 0:
            continue

        num_to_augment = math.ceil(gap / 6)

        # 1. Create a list of all possible image numbers: [1, 2, 3, ..., current_count]
        all_indices = list(range(1, current_count + 1))
        chosen_indices = random.sample(all_indices, num_to_augment)

        # 3. Now loop through the UNIQUE selection
        for idx in chosen_indices:
            parent = "Apple" if "Apple" in category else "Grape"
            image_path = f"./leaves/images/{parent}/{category}/image ({idx}).JPG"

            print(f"Augmenting {image_path}")
            Augmentation.augment_image(image_path)


def main():
    apple_counts = Distribution.run_distribution_analysis(
        './leaves/images/Apple')
    grape_counts = Distribution.run_distribution_analysis(
        './leaves/images/Grape')
    print("Apple Counts:", apple_counts)
    print("Grape Counts:", grape_counts)
    balance_datasets(apple_counts)
    balance_datasets(grape_counts)
    apple_counts = Distribution.run_distribution_analysis(
        './leaves/images/Apple')
    grape_counts = Distribution.run_distribution_analysis(
        './leaves/images/Grape')


if __name__ == "__main__":
    main()
