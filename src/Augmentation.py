import math
import random
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def flip(image):
    return cv2.flip(image, 1)


def rotate(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def blur(image):
    return cv2.GaussianBlur(image, (21, 21), 0)


def crop(image, crop_size=(150, 150)):
    height, width = image.shape[:2]
    max_x = width - crop_size[0]
    max_y = height - crop_size[1]
    start_x = np.random.randint(0, max_x)
    start_y = np.random.randint(0, max_y)
    return image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]


def contrast(image, alpha=1.5, beta=10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def display_images(images):
    for idx, (title, image) in enumerate(images.items()):
        plt.subplot(2, math.ceil(len(images) / 2), idx + 1)
        plt.title(title)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    plt.show()


def save_images(original_filename, images, output_directory='augmented_directory'):
    original_name, original_ext = os.path.splitext(os.path.basename(original_filename))
    for title, image in images.items():
        new_name = f"{original_name}_{title.lower()}{original_ext}"
        cv2.imwrite(f"{output_directory}/{new_name}", image)


def main():
    if len(sys.argv) < 2:
        raise Exception("Please provide an image file")
    filename = sys.argv[1]
    if os.path.exists(filename) is False or os.path.splitext(filename)[1].lower() not in [".jpg", ".jpeg", ".png"]:
        raise Exception("Please provide a valid image file")

    image = cv2.imread(filename)
    augmented_files = {
        "Original": image,
        "Flipped": flip(image),
        "Cropped": crop(image),
        "Contrast": contrast(image),
        "Rotated": rotate(image, random.randint(20, 90)),
        "Blur": blur(image),
    }
    save_images(filename, augmented_files)
    display_images(augmented_files)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)