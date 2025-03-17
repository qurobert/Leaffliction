import argparse
import zipfile
import os
from train_balanced_dataset import balanced_dataset


def argument_parser():
    parser = argparse.ArgumentParser(
        prog="Leaf computer vision to classify plant disease",
        description="This program balanced the dataset, transform it and train with scikit-learn")
    parser.add_argument('dataset', help="Dataset directory")
    parser.add_argument('--output_dir',
                        type=str,
                        default="data/augmented_directory",
                        help="The output directory for training and predict")
    return parser.parse_args()


def save_output_directory_in_zip(output_directory):
    with zipfile.ZipFile(f"train.zip", "w") as zipf:
        for root, dirs, files in os.walk(output_directory):
            for file in files:
                zipf.write(os.path.join(root, file))


if __name__ == "__main__":
    args = argument_parser()
    directory, output_directory = args.dataset, args.output_dir
    try:
        balanced_dataset(directory, output_directory, number_of_features_by_classes=1000)
        save_output_directory_in_zip(output_directory)
    except Exception as e:
        print("Error during balanced dataset : ", e)
