import argparse
import zipfile
import os

import keras

from train_balanced_dataset import balanced_dataset
from keras.src.utils import image_dataset_from_directory
from keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def argument_parser():
    parser = argparse.ArgumentParser(
        prog="Leaf computer vision to classify plant disease",
        description="This program balanced the dataset, transform it and train with scikit-learn")
    parser.add_argument('--raw_dir',
                        type=str,
                        help="Raw dataset directory",
                        default="data/raw")
    parser.add_argument('--processed_dir',
                        type=str,
                        default="data/processed",
                        help="The processed dataset directory")
    parser.add_argument('--no_processing',
                        action="store_false",
                        help="Do not preprocess the dataset")
    return parser.parse_args()


def save_output_directory_in_zip(output_directory):
    with zipfile.ZipFile(f"train.zip", "w") as zipf:
        for root, dirs, files in os.walk(output_directory):
            for file in files:
                zipf.write(os.path.join(root, file))


def split_dataset(directory):
    train_dataset, val_dataset = image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=(256, 256),
        batch_size=32
    )
    return train_dataset, val_dataset


def train_model(train_dataset, val_dataset):
    # data_augmentation_layers = [
    #     layers.RandomFlip("horizontal"),
    #     layers.RandomRotation(0.1),
    # ]
    # for layer in data_augmentation_layers:
    #     images = layer(images)
    # inputs = keras.Input(shape=input_shape)
    # images = layers.Rescaling(1./255)(images)



def main():
    args = argument_parser()
    raw_dir, processed_dir, is_preprocessing = args.raw_dir, args.processed_dir, args.no_processing
    if is_preprocessing:
        balanced_dataset(raw_dir, processed_dir, number_of_features_by_classes=1000)

    # Split the dataset
    train_dataset, val_dataset = split_dataset(processed_dir)

    # Train the model
    model = train_model(train_dataset, val_dataset)
    # Save the output directory in a zip file
    # save_output_directory_in_zip(output_directory)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
