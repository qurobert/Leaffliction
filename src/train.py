import argparse
import zipfile
import os
import keras
from train_balanced_dataset import balanced_dataset
from keras.src.utils import image_dataset_from_directory
from keras import layers
from tensorflow import data as tf_data
import tensorflow as tf
import json

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


def split_dataset_and_prepare_data(directory):
    train_dataset, val_dataset = image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=(256, 256),
        batch_size=32
    )
    # INFO
    class_names = train_dataset.class_names
    with open("./data/model/class_names.json", "w") as f:
        json.dump(class_names, f)
    image_size = (256, 256)

    # Normalize the dataset
    normalization_layer = layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch the dataset for better performance (use the CPU to load the data while the GPU is training)
    train_dataset = train_dataset.prefetch(tf_data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf_data.AUTOTUNE)

    # One hot encoding
    train_dataset = train_dataset.map(lambda x, y: (x, tf.one_hot(y, len(class_names))))
    val_dataset = val_dataset.map(lambda x, y: (x, tf.one_hot(y, len(class_names))))

    return train_dataset, val_dataset, class_names, image_size


def make_model(img_size, num_classes):
    return keras.Sequential([
        keras.Input(shape=(img_size[0], img_size[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Réduit l'overfitting
        layers.Dense(num_classes, activation='softmax')  # Softmax pour la classification multi-classes
    ])


def main():
    args = argument_parser()
    raw_dir, processed_dir, is_preprocessing = args.raw_dir, args.processed_dir, args.no_processing
    if is_preprocessing:
        balanced_dataset(raw_dir, processed_dir, number_of_features_by_classes=1000)

    # Split the dataset and prepare it
    train_dataset, val_dataset, class_names, image_size = split_dataset_and_prepare_data(processed_dir)

    # Make the model
    num_classes = len(class_names)
    model = make_model(image_size, num_classes)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")])

    # Train the model
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=callbacks)

    # TODO: Make a zip of processed + model
    model.save("./data/model/model.keras")
    # save_output_directory_in_zip(output_directory)

    test_loss, test_acc = model.evaluate(val_dataset)
    print(f"Test Accuracy: {test_acc:.2f}")


if __name__ == "__main__":
    # try:
    main()
    # except Exception as e:
    #     print(e)
