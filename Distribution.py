from pathlib import Path
import sys
import matplotlib.pyplot as plt


def analyze_directory(root_path):
    """ Analyzes the given directory and counts the number of files in each subdirectory."""
    root = Path(root_path)

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    counts = {}
    for sd in subdirs:
        file_count = sum(1 for x in sd.iterdir() if x.is_file())
        counts[sd.name] = file_count

    return counts


def make_plots(counts):
    """ Generates a pie chart and bar chart from the counts dictionary."""
    names = list(counts.keys())
    values = list(counts.values())
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'violet']
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(values, autopct='%1.1f%%', colors=colors)

    bars = axs[1].bar(names, values, color=colors)
    axs[1].bar_label(bars, label_type='edge', color='black')

    plt.tight_layout()
    plt.show()


def run_distribution_analysis(path):
    """ Analyzes the distribution of files in the given directory and generates plots."""
    try:
        counts = analyze_directory(path)
        make_plots(counts)
        return counts
    except Exception as e:
        print(f"Error: {e}")


def main():
    if (len(sys.argv) != 2):
        print("Usage: python3 Distribution.py <directory_path>")
        sys.exit(1)
    run_distribution_analysis(sys.argv[1])


if __name__ == "__main__":
    main()
