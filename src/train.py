import argparse
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


if __name__ == "__main__":
    args = argument_parser()
    directory, output_directory = args.dataset, args.output_dir
    try:
        balanced_dataset(directory, output_directory, number_of_features_by_classes=1000)
    except Exception as e:
        print("Error during balanced dataset : ", e)
