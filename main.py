import Distribution
import Augmentation


def balance_datasets(counts):
    max_count = max(counts.values())
    missing_counts = {k: round((max_count - v)/6) for k, v in counts.items()}
    print("Missing Counts for Balance:", missing_counts)
    for category, missing in missing_counts.items():
        for i in range(missing):
            if "Apple" in category:
                image_path = f"./data/Apple/{category}/image ({
                    i+1}).JPG"
            else:
                image_path = f"./data/Grape/{category}/image ({
                    i+1}).JPG"
            print(f"Augmenting {image_path}")
            Augmentation.augment_image(image_path)


def main():
    apple_counts = Distribution.run_distribution_analysis('./data/Apple')
    grape_counts = Distribution.run_distribution_analysis('./data/Grape')
    print("Apple Counts:", apple_counts)
    print("Grape Counts:", grape_counts)
    balance_datasets(apple_counts)
    balance_datasets(grape_counts)
    apple_counts = Distribution.run_distribution_analysis('./data/Apple')
    grape_counts = Distribution.run_distribution_analysis('./data/Grape')


if __name__ == "__main__":
    main()
