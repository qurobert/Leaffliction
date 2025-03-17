from Augmentation import augmented_files, save_images as save_augmented_images
from Distribution import get_directory_files
import cv2
import os
import shutil
import random


def balanced_dataset(directory, output_directory, number_of_features_by_classes=2000):
    dir_files = get_directory_files(directory)
    for key in dir_files.keys():
        count = 0

        # Copy files
        for file in dir_files[key]:
            filename = os.path.join(directory, key, file)
            new_filename = os.path.join(output_directory, key, file)
            if count < number_of_features_by_classes:
                count += 1
                os.makedirs(os.path.dirname(new_filename), exist_ok=True)
                shutil.copy(filename, new_filename)
            else:
                break

        # Augment files if needed
        if count < number_of_features_by_classes:
            random.shuffle(dir_files[key])
            for file in dir_files[key]:
                if count < number_of_features_by_classes:
                    filename = os.path.join(directory, key, file)
                    augmented = augmented_files(filename)
                    original_name, original_ext = os.path.splitext(os.path.basename(filename))
                    for title, image in augmented.items():
                        if count < number_of_features_by_classes:
                            new_name = f"{original_name}_{title.lower()}{original_ext}"
                            cv2.imwrite(f"{output_directory}/{key}/{new_name}", image)
                            count += 1
                        else:
                            break
                else:
                    break
