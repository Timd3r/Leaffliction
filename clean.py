# script that removes all images that dont end with ').JPG'
import os


def clean_directory(root_path):
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            if not (file.endswith(').JPG')):
                file_path = os.path.join(subdir, file)
                os.remove(file_path)


def main():
    clean_directory('./data/Apple')
    clean_directory('./data/Grape')
    print("Removed all augmented files")


if __name__ == "__main__":
    main()
