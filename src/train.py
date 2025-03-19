import argparse
import hashlib
import zipfile
import os
import keras
import logging
from train_balanced_dataset import balanced_dataset
from keras.src.utils import image_dataset_from_directory
from keras import layers
from tensorflow import data as tf_data
import tensorflow as tf
import json


def argument_parser():
    parser = argparse.ArgumentParser(
        prog="Leaf computer vision to classify plant disease",
        description="This program balanced the dataset,"
                    " transform it and train with scikit-learn")
    parser.add_argument('raw_dir',
                        type=str,
                        help="Raw dataset directory",
                        default="data/raw")
    parser.add_argument('--processed_dir',
                        type=str,
                        default="data/processed",
                        help="The processed dataset directory")
    parser.add_argument('--model_dir',
                        type=str,
                        default="data/model",
                        help="The model dataset directory")
    parser.add_argument('--no_processing',
                        action="store_false",
                        help="Do not preprocess the dataset")
    parser.add_argument('--zip_filename',
                        type=str,
                        default="data/model.zip",
                        help="The name of the zip file to save the model")
    return parser.parse_args()


def compute_sha256(file_path):
    """ Calcule le hash SHA256 d'un fichier """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()


def zip_processed_and_model_with_signature(
        processed_dir,
        model_dir,
        zip_path,
        signature_path="signature.txt"):
    with zipfile.ZipFile(zip_path, "w") as zipf:
        # Image training
        for root, dirs, files in os.walk(processed_dir):
            for file in files:
                zipf.write(os.path.join(root, file))

        # Model
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                zipf.write(os.path.join(root, file))

    signature = compute_sha256(zip_path)

    # Ajouter la signature dans signature.txt
    with open(signature_path, "w") as sig_file:
        sig_file.write(f"{signature}  {os.path.basename(zip_path)}\n")

    print(f"✅ ZIP créé : {zip_path}")

    print(f"✅ Signature générée : {signature}")


def split_dataset_and_prepare_data(processed_dir, model_dir):
    train_dataset, val_dataset = image_dataset_from_directory(
        processed_dir,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=(256, 256),
        batch_size=32
    )
    # INFO
    class_names = train_dataset.class_names
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "class_names.json"), "w+") as f:
        json.dump(class_names, f)
    image_size = (256, 256)

    # Normalize the dataset
    normalization_layer = layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch the dataset for better performance
    # (use the CPU to load the data while the GPU is training)
    train_dataset = train_dataset.prefetch(tf_data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf_data.AUTOTUNE)

    # One hot encoding
    train_dataset = train_dataset.map(
        lambda x, y: (x, tf.one_hot(y, len(class_names))))
    val_dataset = val_dataset.map(
        lambda x, y: (x, tf.one_hot(y, len(class_names))))

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
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])


def configure_logger():
    log_filename = "training.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    args = argument_parser()
    raw_dir = args.raw_dir
    processed_dir = args.processed_dir
    is_preprocessing = args.no_processing
    model_dir = args.model_dir
    zip_filename = args.zip_filename

    # Configure the logger
    configure_logger()

    # Preprocess the dataset
    # if is_preprocessing:
    #     balanced_dataset(raw_dir,
    #                      processed_dir,
    #                      number_of_features_by_classes=1000)

    # Split the dataset and prepare it
    (train_dataset,
     val_dataset,
     class_names,
     image_size) = split_dataset_and_prepare_data(processed_dir, model_dir)

    # Make the model
    num_classes = len(class_names)
    model = make_model(image_size, num_classes)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")])

    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    # Save the model
    model.save(os.path.join(model_dir, "model.keras"))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(val_dataset)
    log_message = f"Test Accuracy: {test_acc:.2f}"
    print(log_message)
    logging.info(log_message)

    # Make a zip of processed + model and remove directory
    zip_processed_and_model_with_signature(processed_dir,
                                           model_dir,
                                           zip_filename)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
