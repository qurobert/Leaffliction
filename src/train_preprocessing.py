import argparse
from types import SimpleNamespace
from Augmentation import augmented_files
from Distribution import get_directory_files
from Transformation import process_image
import tqdm
import cv2
import os
import shutil
import random
import numpy as np


def balanced_dataset(raw_directory,
                     processed_directory,
                     number_of_features_by_classes=1500):
    dir_files = get_directory_files(raw_directory)
    os.makedirs(processed_directory, exist_ok=True)
    if len(os.listdir(processed_directory)) != 0:
        shutil.rmtree(processed_directory)
        os.makedirs(processed_directory, exist_ok=True)
    for key in tqdm.tqdm(dir_files.keys(), "Balancing dataset"):
        count = 0

        # Copy files
        for file in dir_files[key]:
            filename = str(os.path.join(raw_directory, key, file))
            new_filename = str(os.path.join(processed_directory, key, file))

            if count < number_of_features_by_classes:
                count += 1
                os.makedirs(os.path.dirname(new_filename), exist_ok=True)
                shutil.copy(filename, new_filename)
            else:
                break

        # Augment files if needed
        error_count = 0
        while count < number_of_features_by_classes:
            if number_of_features_by_classes - count > len(dir_files[key]) * 6:
                raise Exception("Not enough files to augment, "
                                "please add more files to the dataset")
            # Get random file
            file = random.choice(list(dir_files[key]))
            filename = os.path.join(raw_directory, key, file)

            # Get random augmentation
            augmented = augmented_files(filename)
            augmented.pop("Original")
            # augmented.pop("Cropped")
            # augmented.pop("Rotated")
            # augmented.pop("Blur")
            # augmented.pop("Flipped")
            # augmented.pop("Contrast")
            # augmented.pop("Projective")

            title = random.choice(list(augmented.keys()))

            # Save augmented image
            base_name = os.path.basename(filename)
            original_name, original_ext = os.path.splitext(base_name)
            new_name = f"{original_name}_{title.lower()}{original_ext}"
            if new_name not in os.listdir(f"{processed_directory}/{key}"):
                cv2.imwrite(f"{processed_directory}/{key}/{new_name}",
                            augmented[title])

                # Update count
                count += 1
            else:
                error_count += 1
                if error_count > 10000:
                    raise Exception("Error count is too high, "
                                    "please check the code")


def remove_background(processed_dir, mask_directory):
    if os.path.exists(mask_directory):
        shutil.rmtree(mask_directory)
    os.makedirs(mask_directory, exist_ok=True)
    for root, _, files in os.walk(processed_dir):
        for file in tqdm.tqdm(files,  f"Removing background directory {root}"):
            filename = os.path.join(root, file)
            dir_filename = os.path.basename(os.path.dirname(filename))
            # destination = os.path.join(mask_directory, dir_filename)
            # os.makedirs(augmented_directory, exist_ok=True)
            # remove_background_cv2_part(filename, dir_filename)
            process_image(filename, os.path.join(mask_directory, dir_filename), ["masked"])


def remove_background_cv2_part(filename, augmented_directory, save_image=True):
    image = cv2.imread(filename)

    # Convertir l'image en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les seuils pour la couleur verte
    lower_green = np.array([35, 40, 40])  # Borne inférieure (H, S, V)
    upper_green = np.array([85, 255, 255])  # Borne supérieure (H, S, V)

    # Créer un masque pour la couleur verte
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Améliorer le masque (ouverture + fermeture pour enlever le bruit)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Inverser le masque pour ne garder que la feuille
    mask_inv = cv2.bitwise_not(mask)

    # Appliquer le masque pour supprimer le fond gris
    result = cv2.bitwise_and(image, image, mask=mask)

    # RGB
    # result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # print(result.shape)

    # Enregistrer l'image
    if save_image:
        result_filename = os.path.join(augmented_directory, os.path.splitext(os.path.basename(filename))[0] + "_masked.JPG")
        # result_filename = os.path.join(destination, os.path.basename(filename))
        cv2.imwrite(str(result_filename), result)
    return result
