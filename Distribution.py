from pathlib import Path
import sys
import matplotlib.pyplot as plt


def analyze_directory(root_path):
    root = Path(root_path)

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    counts = {}
    for sd in subdirs:
        file_count = sum(1 for x in sd.iterdir() if x.is_file())
        counts[sd.name] = file_count

    return counts


def make_plots(counts):
    names = list(counts.keys())
    values = list(counts.values())
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'violet']
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(values, autopct='%1.1f%%', colors=colors)

    axs[1].bar(names, values, color=colors)
    plt.tight_layout()
    plt.show()


def main():
    if (len(sys.argv) != 2):
        print("Usage: python3 Distribution.py <directory_path>")
        sys.exit(1)

    counts = analyze_directory(sys.argv[1])
    make_plots(counts)


if __name__ == "__main__":
    main()
