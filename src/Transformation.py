#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import matplotlib
import math
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def parse_arguments():
    parser = argparse.ArgumentParser(description='Leaf Image Transformation Tool')
    parser.add_argument('path', nargs='?', help='Path to a single image or source directory')
    parser.add_argument('-src', '--source', help='Source directory containing images')
    parser.add_argument('-dst', '--destination', help='Destination directory for saving transformed images')
    parser.add_argument('-mask', action='store_true', help='Apply masking operations')

    return parser.parse_args()


# def create_color_histogram(img_rgb):
#     plt.figure(figsize=(10, 6))
#     plt.grid(True, alpha=0.3)
#
#     colors = ['red', 'green', 'blue']
#     for i, color in enumerate(colors):
#         hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
#         hist = hist / img_rgb.size * 100  # Convert to percentage
#         plt.plot(hist, color=color, label=color)
#
#     img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
#
#     # Hue histogram
#     hue_hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
#     hue_hist = hue_hist / img_rgb.size * 100  # Convert to percentage
#     plt.plot(np.interp(np.arange(256), np.linspace(0, 255, 180), hue_hist.flatten()),
#              color='purple', label='hue')
#
#     # Saturation histogram
#     sat_hist = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])
#     sat_hist = sat_hist / img_rgb.size * 100  # Convert to percentage
#     plt.plot(sat_hist, color='cyan', label='saturation')
#
#     # Value histogram
#     val_hist = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])
#     val_hist = val_hist / img_rgb.size * 100  # Convert to percentage
#     plt.plot(val_hist, color='orange', label='value')
#
#     # Convert to LAB for lightness
#     img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
#     light_hist = cv2.calcHist([img_lab], [0], None, [256], [0, 256])
#     light_hist = light_hist / img_rgb.size * 100  # Convert to percentage
#     plt.plot(light_hist, color='gray', label='lightness')
#
#     # Green-Magenta channel (from LAB b channel)
#     gm_hist = cv2.calcHist([img_lab], [2], None, [256], [0, 256])
#     gm_hist = gm_hist / img_rgb.size * 100  # Convert to percentage
#     plt.plot(gm_hist, color='magenta', label='green-magenta')
#
#     # Blue-Yellow channel (from LAB a channel)
#     by_hist = cv2.calcHist([img_lab], [1], None, [256], [0, 256])
#     by_hist = by_hist / img_rgb.size * 100  # Convert to percentage
#     plt.plot(by_hist, color='yellow', label='blue-yellow')
#
#     plt.xlabel('Pixel intensity')
#     plt.ylabel('Proportion of pixels (%)')
#     plt.legend(title='color Channel', loc='upper right', bbox_to_anchor=(1.25, 1))
#     plt.ylim(0, 10)  # Set y-axis limit to 10%
#     plt.title("Figure IV.7: Color histogram")
#
#     plt.tight_layout()
#     fig = plt.gcf()
#     fig.canvas.draw()
#     hist_img = np.array(fig.canvas.renderer.buffer_rgba())
#
#     plt.close()
#
#     return hist_img

def create_color_histogram(img_rgb):
    """Create a color histogram visualization from an RGB image"""
    # Create a figure for the histogram
    fig = plt.figure(figsize=(8, 6))

    # Get the histogram data for each channel
    colors = ('red', 'green', 'blue')
    color_labels = ['red', 'blue-yellow', 'green', 'green-magenta', 'hue', 'lightness', 'red', 'saturation', 'value']
    color_values = ['r', 'c', 'g', 'm', 'violet', 'gray', 'r', 'y', 'b']

    # RGB histograms
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color_values[i], alpha=0.7, label=color_labels[i])

    # Convert to HSV for additional histograms
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # HSV histograms
    for i, label in enumerate(['hue', 'saturation', 'value']):
        if i == 0:  # Hue has range 0-179 in OpenCV
            hist = cv2.calcHist([img_hsv], [i], None, [180], [0, 180])
            plt.plot(hist, color=color_values[i + 4], alpha=0.7, label=color_labels[i + 4])
        else:
            hist = cv2.calcHist([img_hsv], [i], None, [256], [0, 256])
            plt.plot(hist, color=color_values[i + 6], alpha=0.7, label=color_labels[i + 6])

    # LAB color space (for additional histograms)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # L (lightness) histogram
    hist = cv2.calcHist([img_lab], [0], None, [256], [0, 256])
    plt.plot(hist, color=color_values[5], alpha=0.7, label=color_labels[5])

    # a (green-magenta) histogram
    hist = cv2.calcHist([img_lab], [1], None, [256], [0, 256])
    plt.plot(hist, color=color_values[3], alpha=0.7, label=color_labels[3])

    # b (blue-yellow) histogram
    hist = cv2.calcHist([img_lab], [2], None, [256], [0, 256])
    plt.plot(hist, color=color_values[1], alpha=0.7, label=color_labels[1])

    plt.xlabel('Pixel intensity')
    plt.ylabel('Proportion of pixels (%)')
    plt.title('Color Histogram')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(alpha=0.3)

    # Convert the plot to an image
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    hist_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    hist_img = hist_img.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return hist_img


def process_image(image_path, destination=None, apply_mask=False):
    """Process a single image with various transformations using PlantCV"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image: {image_path}")
        return

    # Create a list to store images and their titles
    transformations = []

    # 1. Original image
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformations.append((img_rgb, "Original"))

    # Create a PlantCV parameters class
    pcv_params = pcv.Params()

    # 2. Gaussian blur - using PlantCV's wrapper
    gaussian_blur = pcv.gaussian_blur(img_rgb, ksize=(5, 5))
    transformations.append((gaussian_blur, "Gaussian blur"))

    # 3. Create mask with PlantCV
    # Convert to HSV color space
    s = pcv.rgb2gray_hsv(img_rgb, 's')
    # Threshold the saturation channel
    s_thresh = pcv.threshold.binary(s, 80, 'light')
    # Median Blur
    s_mblur = pcv.median_blur(s_thresh, 5)
    # Create a mask using the LAB color space for better plant segmentation
    # Convert RGB to LAB and extract the Blue channel
    b = pcv.rgb2gray_lab(img_rgb, 'b')
    # Threshold the blue image
    b_thresh = pcv.threshold.binary(b, 160, 'light')
    # Fill small objects
    b_fill = pcv.fill(b_thresh, 10)
    # Combine the thresholds
    mask = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_fill)
    # Analyze object (ROI)
    roi = pcv.roi.rectangle(img=img_rgb, x=100, y=100, h=100, w=100)
    mask_roi = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
    transformations.append((mask_roi, "ROI"))

    # 4. Apply Mask to the image
    # Apply the mask to the original image
    roi = pcv.apply_mask(img=img_rgb, mask=mask_roi, mask_color='white')
    transformations.append((roi, "Applied Mask"))

    # 5. Analyze image
    shape_img = pcv.analyze.size(img=img_rgb, labeled_mask=mask_roi, n_labels=1)
    pcv.outputs.save_results(filename="results.txt", outformat="json")
    transformations.append((shape_img, "Analyze Object"))


    # 6. Pseudolandmarks
    pseudo_img = img_rgb.copy()
    pcv.params.debug_outdir = "./temp/"
    top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(img=pseudo_img, mask=mask_roi, label="default")
    # Draw top pseudolandmarks in red
    for point in top_x:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(pseudo_img, (x, y), 3, (255, 0, 0), -1)
    # Draw center pseudolandmarks in green
    for point in center_v_x:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(pseudo_img, (x, y), 3, (0, 255, 0), -1)
    # Draw bottom pseudolandmarks in blue
    for point in bottom_x:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(pseudo_img, (x, y), 3, (0, 0, 255), -1)
    transformations.append((pseudo_img, "Pseudolandmarks"))


    # 7. Create and add color histogram
    # Compute the histograms for each channel
    hist_img = create_color_histogram(img_rgb)
    transformations.append((hist_img, "Color Histogram"))

    # Display or save the transformations
    if destination:
        # Save all transformations to the destination folder
        base_name = os.path.basename(image_path)
        file_name, ext = os.path.splitext(base_name)

        for img_trans, title in transformations:
            save_path = os.path.join(destination, f"{file_name}_{title.lower().replace(' ', '_')}{ext}")
            # Convert back to BGR for saving with OpenCV
            if len(img_trans.shape) == 3 and img_trans.shape[2] == 3:
                img_save = cv2.cvtColor(img_trans, cv2.COLOR_RGB2BGR)
            else:
                img_save = img_trans
            cv2.imwrite(save_path, img_save)
    else:
        # Display all transformations
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

        for ax in axes[7:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return transformations


def main():
    args = parse_arguments()

    # Create destination directory if specified and doesn't exist
    if args.destination and not os.path.exists(args.destination):
        os.makedirs(args.destination)

    if args.path and os.path.isfile(args.path):
        process_image(args.path, args.destination, args.mask)

    elif args.source and os.path.isdir(args.source):
        if not args.destination:
            print("Error: Destination directory (-dst) is required when processing a directory")
            return

        for filename in os.listdir(args.source):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(args.source, filename)
                process_image(image_path, args.destination, args.mask)

    else:
        print("Error: Please provide a valid image path or source directory")


if __name__ == "__main__":
    main()
