import argparse
import subprocess
from Augmentation import augmented_files, save_images as save_augmented_images
from Distribution import get_directory_files
import cv2
import os
import shutil


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


def save_file(filename, output_directory):
    path = os.path.join(output_directory, filename)
    cv2.imwrite(path)


def balanced_dataset(directory, output_directory, number_of_features_by_classes=2000):
    dir_files = get_directory_files(directory)
    print(output_directory)
    for key in dir_files.keys():
        count = 0
        for file in dir_files[key]:
            filename = os.path.join(directory, key, file)
            shutil.copy(filename, os.path.join(output_directory, file))
            break;
        break


if __name__ == "__main__":
    args = argument_parser()
    directory = args.dataset
    output_directory = args.output_dir
    try:
        balanced_dataset(directory, output_directory)
    except Exception as e:
        print("Error during balanced dataset : ", e)
