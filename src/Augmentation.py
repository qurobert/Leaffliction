import math
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


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
    cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]
    pad_top = (height - crop_size[1]) // 2
    pad_bottom = height - crop_size[1] - pad_top
    pad_left = (width - crop_size[0]) // 2
    pad_right = width - crop_size[0] - pad_left

    padded_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image


def contrast(image, alpha=1.5, beta=10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def projective(image):
    h, w, _ = image.shape

    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])

    dst_pts = np.float32([
        [w * 0.1, h * 0.2],
        [w * 0.9, h * 0.1],
        [w * 0.2, h * 0.9],
        [w * 0.8, h * 0.8]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    projected_img = cv2.warpPerspective(image, M, (w, h))

    return projected_img


def argument_parser():
    parser = argparse.ArgumentParser(
        prog="Augmentation leaf image",
        description="This program augment the dataset")
    parser.add_argument('filename', help="Filename to augment")
    parser.add_argument('--no-display', action='store_false', help="Display augmentation")
    args = parser.parse_args()
    filename = args.filename
    if os.path.exists(filename) is False or os.path.splitext(filename)[1].lower() not in [".jpg", ".jpeg", ".png"]:
        raise Exception("Please provide a valid image file")
    return [filename, args.no_display]


def display_images(images):
    for idx, (title, image) in enumerate(images.items()):
        plt.subplot(2, math.ceil(len(images) / 2), idx + 1)
        plt.title(title)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    plt.show()


def augmented_files(filename):
    image = cv2.imread(filename)
    return {
        "Original": image,
        "Flipped": flip(image),
        "Cropped": crop(image),
        "Contrast": contrast(image),
        "Rotated": rotate(image, random.randint(20, 90)),
        "Blur": blur(image),
        "Projective": projective(image)
    }


def save_images(original_filename, images, output_directory='augmented_directory'):
    original_name, original_ext = os.path.splitext(os.path.basename(original_filename))
    os.makedirs(output_directory, exist_ok=True)
    for title, image in images.items():
        new_name = f"{original_name}_{title.lower()}{original_ext}"
        cv2.imwrite(f"{output_directory}/{new_name}", image)


def main():
    filename, is_display = argument_parser()
    images = augmented_files(filename)
    save_images(filename, images)
    if is_display:
        display_images(images)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
