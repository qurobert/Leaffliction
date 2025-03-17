from Augmentation import augmented_files
from Distribution import get_directory_files
import tqdm
import cv2
import os
import shutil
import random


def balanced_dataset(directory, output_directory, number_of_features_by_classes=2000):
    dir_files = get_directory_files(directory)
    os.makedirs(output_directory, exist_ok=True)
    if len(os.listdir(output_directory)) != 0:
        raise Exception("The processed directory is not empty, please remove it before running this script or run the "
                        "code with this option : --no_processing to run the code without preprocessing the dataset")
    for key in tqdm.tqdm(dir_files.keys(), "Balancing dataset"):
        count = 0

        # Copy files
        for file in dir_files[key]:
            filename = str(os.path.join(directory, key, file))
            new_filename = str(os.path.join(output_directory, key, file))

            if count < number_of_features_by_classes:
                count += 1
                os.makedirs(os.path.dirname(new_filename), exist_ok=True)
                shutil.copy(filename, new_filename)
            else:
                break

        # Augment files if needed
        error_count = 0
        while count < number_of_features_by_classes:
            # Get random file
            file = random.choice(list(dir_files[key]))
            filename = os.path.join(directory, key, file)

            # Get random augmentation
            augmented = augmented_files(filename)
            augmented.pop("Original")
            title = random.choice(list(augmented.keys()))

            # Save augmented image
            original_name, original_ext = os.path.splitext(os.path.basename(filename))
            new_name = f"{original_name}_{title.lower()}{original_ext}"
            if new_name not in os.listdir(f"{output_directory}/{key}"):
                cv2.imwrite(f"{output_directory}/{key}/{new_name}", augmented[title])

                # Update count
                count += 1
            else:
                error_count += 1
                if error_count > 1000:
                    raise Exception("Error count is too high, please check the code")
