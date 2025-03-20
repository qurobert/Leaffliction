# !/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import json


def load_threshold_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load threshold config file: {e}")
        return {}


def load_image(image_path):
    img, path, filename = pcv.readimage(image_path)
    if img is None:
        print(f"Error: Could not read the image: {image_path}")
        return None, None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, path, filename


def apply_gaussian_blur(img_rgb):
    s_gblur = pcv.gaussian_blur(img=img_rgb, ksize=(5, 5),
                                sigma_x=0, sigma_y=None)

    return s_gblur


def create_mask(img_rgb, image_path=None, threshold_method='auto', threshold_value=130, config_path=None):
    s = pcv.rgb2gray_hsv(rgb_img=img_rgb, channel="s")
    s_thresh = pcv.threshold.binary(
        gray_img=s, threshold=60, object_type="light"
    )
    s_gblur = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5),
                                sigma_x=0, sigma_y=None)

    b = pcv.rgb2gray_lab(rgb_img=img_rgb, channel="b")
    b_thresh = pcv.threshold.binary(
        gray_img=b, threshold=200, object_type="light"
    )
    bs = pcv.logical_or(bin_img1=s_gblur, bin_img2=b_thresh)

    return img_rgb, bs


def apply_mask_to_image(masked_img, mask):
    return pcv.apply_mask(img=masked_img, mask=mask, mask_color="white")


def analyze_object(img_rgb, mask):
    shape_img = pcv.analyze.size(img=img_rgb, labeled_mask=mask, n_labels=1)
    pcv.outputs.save_results(filename="results.txt", outformat="json")
    return shape_img


def generate_pseudolandmarks(img_rgb, mask):
    pseudo_img = img_rgb.copy()
    pcv.params.debug_outdir = "./temp/"
    top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(img=pseudo_img, mask=mask, label="default")

    for point in top_x:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(pseudo_img, (x, y), 3, (255, 0, 0), -1)
    for point in center_v_x:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(pseudo_img, (x, y), 3, (0, 255, 0), -1)
    for point in bottom_x:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(pseudo_img, (x, y), 3, (0, 0, 255), -1)

    return pseudo_img


def create_color_histogram(img_rgb):
    fig = plt.figure(figsize=(8, 6))

    colors = ('red', 'green', 'blue')
    color_labels = ['red', 'blue-yellow', 'green', 'green-magenta', 'hue', 'lightness', 'red', 'saturation', 'value']
    color_values = ['r', 'c', 'g', 'm', 'violet', 'gray', 'r', 'y', 'b']

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color_values[i], alpha=0.7, label=color_labels[i])

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    for i, label in enumerate(['hue', 'saturation', 'value']):
        if i == 0:  # Hue has range 0-179 in OpenCV
            hist = cv2.calcHist([img_hsv], [i], None, [180], [0, 180])
            plt.plot(hist, color=color_values[i + 4], alpha=0.7, label=color_labels[i + 4])
        else:
            hist = cv2.calcHist([img_hsv], [i], None, [256], [0, 256])
            plt.plot(hist, color=color_values[i + 6], alpha=0.7, label=color_labels[i + 6])

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    hist = cv2.calcHist([img_lab], [0], None, [256], [0, 256])
    plt.plot(hist, color=color_values[5], alpha=0.7, label=color_labels[5])

    hist = cv2.calcHist([img_lab], [1], None, [256], [0, 256])
    plt.plot(hist, color=color_values[3], alpha=0.7, label=color_labels[3])

    hist = cv2.calcHist([img_lab], [2], None, [256], [0, 256])
    plt.plot(hist, color=color_values[1], alpha=0.7, label=color_labels[1])

    plt.xlabel('Pixel intensity')
    plt.ylabel('Proportion of pixels (%)')
    plt.title('Color Histogram')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(alpha=0.3)

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    hist_img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    hist_img = cv2.cvtColor(hist_img, cv2.COLOR_RGBA2RGB)

    plt.close()

    return hist_img


def save_image(img, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_save = img
    cv2.imwrite(save_path, img_save)


def display_images(transformations):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, (img_trans, title) in enumerate(transformations):
        if i < len(axes):
            if len(img_trans.shape) == 2:  # Grayscale image
                axes[i].imshow(img_trans, cmap='gray')
            else:  # Color image
                axes[i].imshow(img_trans)
            axes[i].set_title(title)
            axes[i].axis('off')

    for ax in axes[len(transformations):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def process_image(image_path, destination=None, operations=None, display=False,
                  threshold_method='auto', threshold_value=130, config_path=None, apply_blur=True):
    img_rgb, path, filename = load_image(image_path)
    if img_rgb is None:
        return None

    if operations is None:
        operations = ['mask']

    results = {}
    transformations = []  # For display purposes

    base_name = os.path.basename(image_path)
    file_name, ext = os.path.splitext(base_name)

    if 'original' in operations:
        results['original'] = img_rgb
        transformations.append((img_rgb, "Original"))
        if destination:
            save_path = os.path.join(destination, f"{file_name}_original{ext}")
            save_image(img_rgb, save_path)

    if apply_blur or 'blur' in operations:
        img_rgb_processed = apply_gaussian_blur(img_rgb)
        if 'blur' in operations:
            results['blur'] = img_rgb_processed
            transformations.append((img_rgb_processed, "Gaussian blur"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_gaussian_blur{ext}")
                save_image(img_rgb_processed, save_path)

    if any(op in operations for op in ['mask', 'masked', 'analyze', 'landmarks']):
        img_rbg, mask = create_mask(img_rgb, image_path, threshold_method, threshold_value, config_path)
        if 'mask' in operations:
            results['mask'] = mask
            transformations.append((mask, "Mask"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_mask{ext}")
                save_image(mask, save_path)

        if 'masked' in operations:
            applied = apply_mask_to_image(img_rbg, mask)
            results['masked'] = applied
            transformations.append((applied, "Applied Mask"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_applied_mask{ext}")
                save_image(applied, save_path)

        if 'analyze' in operations:
            shape_img = analyze_object(img_rgb, mask)
            results['analyze'] = shape_img
            transformations.append((shape_img, "Analyze Object"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_analyze_object{ext}")
                save_image(shape_img, save_path)

        if 'landmarks' in operations:
            pseudo_img = generate_pseudolandmarks(img_rgb, mask)
            results['landmarks'] = pseudo_img
            transformations.append((pseudo_img, "Pseudolandmarks"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_pseudolandmarks{ext}")
                save_image(pseudo_img, save_path)

    if 'histogram' in operations:
        hist_img = create_color_histogram(img_rgb)
        results['histogram'] = hist_img
        transformations.append((hist_img, "Color Histogram"))
        if destination:
            save_path = os.path.join(destination, f"{file_name}_color_histogram{ext}")
            save_image(hist_img, save_path)

    if display and transformations:
        display_images(transformations)

    return results


def process_directory(source_dir, destination_dir, operations=None,
                      threshold_method='auto', threshold_value=130, config_path=None, apply_blur=True):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    results = {}
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(source_dir, filename)
            print(f"Processing {image_path}...")
            img_results = process_image(
                image_path,
                destination_dir,
                operations,
                False,
                threshold_method,
                threshold_value,
                config_path,
                apply_blur
            )
            if img_results:
                results[filename] = img_results

    return results


def main():
    parser = argparse.ArgumentParser(description='Leaf Image Transformation Tool')
    parser.add_argument('path', nargs='?', help='Path to a single image to display transformations')
    parser.add_argument('-src', '--source', help='Source directory containing images to process and save')
    parser.add_argument('-dst', '--destination', help='Destination directory for saving transformed images')

    parser.add_argument('-original', action='store_true', help='Save original image')
    parser.add_argument('-blur', action='store_true', help='Apply Gaussian blur transformation')
    parser.add_argument('-mask', action='store_true', help='Generate binary mask')
    parser.add_argument('-masked', action='store_true', help='Apply mask to the original image')
    parser.add_argument('-analyze', action='store_true', help='Analyze object size and shape')
    parser.add_argument('-landmarks', action='store_true', help='Generate pseudolandmarks')
    parser.add_argument('-histogram', action='store_true', help='Generate color histogram')
    parser.add_argument('-all', action='store_true', help='Apply all transformations')

    parser.add_argument('-threshold-method', choices=['auto', 'light', 'dark'], default='auto',
                        help='Thresholding method (auto determines based on disease in filename)')
    parser.add_argument('-threshold-value', type=int, default=130,
                        help='Threshold value for LAB channel')
    parser.add_argument('-config', help='Path to JSON config file with threshold settings for each disease')

    args = parser.parse_args()

    operations = []
    if args.original or args.all:
        operations.append('original')
    if args.blur or args.all:
        operations.append('blur')
    if args.mask or args.all:
        operations.append('mask')
    if args.masked or args.all:
        operations.append('masked')
    if args.analyze or args.all:
        operations.append('analyze')
    if args.landmarks or args.all:
        operations.append('landmarks')
    if args.histogram or args.all:
        operations.append('histogram')

    if not operations and not args.all:
        operations = ['original', 'blur', 'mask', 'masked', 'analyze', 'landmarks', 'histogram']

    if args.path and os.path.isfile(args.path):
        print(f"Processing single image: {args.path}")
        process_image(
            args.path,
            args.destination,
            operations,
            True,  # display=True
            args.threshold_method,
            args.threshold_value,
            args.config
        )

    elif args.source and os.path.isdir(args.source):
        if not args.destination:
            print("Error: Destination directory (-dst) is required when processing a directory")
            return

        print(f"Processing all images in directory: {args.source}")
        print(f"Saving transformations to: {args.destination}")

        process_directory(
            args.source,
            args.destination,
            operations,
            args.threshold_method,
            args.threshold_value,
            args.config
        )

    else:
        print("Error: Please provide a valid image path or source directory")
        parser.print_help()


if __name__ == "__main__":
    main()