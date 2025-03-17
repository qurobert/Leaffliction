#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import matplotlib

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


def create_color_histogram(img_rgb):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)

    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        hist = hist / img_rgb.size * 100  # Convert to percentage
        plt.plot(hist, color=color, label=color)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Hue histogram
    hue_hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
    hue_hist = hue_hist / img_rgb.size * 100  # Convert to percentage
    plt.plot(np.interp(np.arange(256), np.linspace(0, 255, 180), hue_hist.flatten()),
             color='purple', label='hue')

    # Saturation histogram
    sat_hist = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])
    sat_hist = sat_hist / img_rgb.size * 100  # Convert to percentage
    plt.plot(sat_hist, color='cyan', label='saturation')

    # Value histogram
    val_hist = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])
    val_hist = val_hist / img_rgb.size * 100  # Convert to percentage
    plt.plot(val_hist, color='orange', label='value')

    # Convert to LAB for lightness
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    light_hist = cv2.calcHist([img_lab], [0], None, [256], [0, 256])
    light_hist = light_hist / img_rgb.size * 100  # Convert to percentage
    plt.plot(light_hist, color='gray', label='lightness')

    # Green-Magenta channel (from LAB b channel)
    gm_hist = cv2.calcHist([img_lab], [2], None, [256], [0, 256])
    gm_hist = gm_hist / img_rgb.size * 100  # Convert to percentage
    plt.plot(gm_hist, color='magenta', label='green-magenta')

    # Blue-Yellow channel (from LAB a channel)
    by_hist = cv2.calcHist([img_lab], [1], None, [256], [0, 256])
    by_hist = by_hist / img_rgb.size * 100  # Convert to percentage
    plt.plot(by_hist, color='yellow', label='blue-yellow')

    plt.xlabel('Pixel intensity')
    plt.ylabel('Proportion of pixels (%)')
    plt.legend(title='color Channel', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.ylim(0, 10)  # Set y-axis limit to 10%
    plt.title("Figure IV.7: Color histogram")

    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    hist_img = np.array(fig.canvas.renderer.buffer_rgba())

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
    transformations.append((img, "Original"))

    # Create a PlantCV parameters class
    pcv_params = pcv.Params()

    # 2. Gaussian blur - using PlantCV's wrapper
    gaussian_blur = pcv.gaussian_blur(img_rgb, ksize=(5, 5), sigma_x=0, sigma_y=None)
    transformations.append((gaussian_blur, "Gaussian blur"))

    # 3. Create mask with PlantCV
    # Convert to HSV color space
    s = pcv.rgb2gray_hsv(rgb_img=img_rgb, channel='s')

    # Threshold the saturation channel
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=30, object_type='light')

    # Create a mask using the LAB color space for better plant segmentation
    b = pcv.rgb2gray_lab(rgb_img=img_rgb, channel='b')
    b_thresh = pcv.threshold.binary(gray_img=b, threshold=100, object_type='light')

    # Combine the thresholds
    mask = pcv.logical_and(bin_img1=s_thresh, bin_img2=b_thresh)

    # Fill small holes
    mask_filled = pcv.fill(bin_img=mask, size=50)

    # Apply median blur to smooth the mask
    mask_blur = pcv.median_blur(gray_img=mask_filled, ksize=5)
    transformations.append((mask_blur, "Mask"))

    # 4. ROI objects
    # Apply the mask to the original image
    roi = pcv.apply_mask(img=img_rgb, mask=mask_blur, mask_color='black')
    transformations.append((roi, "ROI objects"))

    # # 5. Analyze object (find contours)
    # # Find objects (contours)
    # contours, hierarchy = pcv.find_objects(img=img_rgb, mask=mask_blur)
    #
    # # If contours are found
    # if len(contours) > 0:
    #     # Select the largest contour
    #     obj, mask_obj = pcv.object_composition(img=img_rgb, contours=contours, hierarchy=hierarchy)
    #
    #     # Draw the contour on the image
    #     analyzed_img = np.copy(img_rgb)
    #     analyzed_img = pcv.visualize.draw_contours(analyzed_img, obj, (0, 255, 0))
    #     transformations.append((analyzed_img, "Analyze object"))
    #
    #     # 6. Generate pseudolandmarks
    #     # Set landmark points parameters
    #     landmarks_img = np.copy(img_rgb)
    #
    #     # Use PlantCV's landmark analysis to find boundary points
    #     landmarks = pcv.landmark_reference_pt_dist(img=img_rgb, mask=mask_obj,
    #                                                points=30, normalize=True)
    #
    #     # Draw landmark points on the image
    #     landmarks_img = pcv.visualize.draw_landmark_points(landmarks_img, landmarks)
    #     transformations.append((landmarks_img, "Pseudolandmarks"))
    #
    #     # 7. Color histogram using PlantCV
    #     # Create histogram of the original RGB image with the mask applied
    #     hist_fig, hist_data = pcv.visualize.colorspaces(rgb_img=img_rgb, mask=mask_obj)
    #
    #     # Convert matplotlib figure to numpy array for display/saving
    #     hist_fig.canvas.draw()
    #     hist_img = np.array(hist_fig.canvas.renderer.buffer_rgba())
    #
    #     # Close the histogram figure
    #     plt.close(hist_fig)
    #
    #     transformations.append((hist_img, "Color histogram"))
    #
    # else:
    #     # If no objects found, add placeholders
    #     transformations.append((img_rgb.copy(), "No objects found for analysis"))
    #     transformations.append((img_rgb.copy(), "No objects found for pseudolandmarks"))
    #     transformations.append((img_rgb.copy(), "No objects found for color histogram"))

    # Display or save the transformations
    if destination:
        # Save the transformations to the destination directory
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        for i, (img, title) in enumerate(transformations):
            save_path = os.path.join(destination, f"{name}_{i + 1}_{title.replace(' ', '_')}{ext}")
            # Check if image is grayscale or has alpha channel
            if len(img.shape) == 2:  # Grayscale
                cv2.imwrite(save_path, img)
            elif img.shape[2] == 4:  # RGBA (from matplotlib figure)
                # Convert RGBA to RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                # Convert RGB to BGR for OpenCV
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_img)
            else:  # RGB
                # Convert RGB to BGR for OpenCV
                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_img)
        print(f"Saved transformations for {filename} to {destination}")
    else:
        # Display the transformations
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, (img, title) in enumerate(transformations):
            if i < len(axes):
                if len(img.shape) == 2:  # If grayscale
                    axes[i].imshow(img, cmap='gray')
                elif img.shape[2] == 4:  # RGBA (from matplotlib)
                    # Convert RGBA to RGB
                    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
                else:  # RGB
                    axes[i].imshow(img)
                axes[i].set_title(f"Figure IV.{i + 1}: {title}")
                axes[i].axis('off')

        # Hide any unused subplots
        for i in range(len(transformations), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display figure: {e}")
            output_file = f"output_{os.path.basename(image_path)}.png"
            plt.savefig(output_file)
            plt.close()
            print(f"Transformations saved to {output_file}")


# def process_image(image_path, destination=None, apply_mask=False):
#
#     # Read in the garyscale data
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not read the image: {image_path}")
#         return
#
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     transformations = []
#
#     transformations.append((img_rgb, "Original"))
#
#     gaussian_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
#     transformations.append((gaussian_blur, "Gaussian blur"))
#
#     hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
#     s = hsv[:, :, 1]
#     _, mask = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)
#     transformations.append((mask, "Mask"))
#
#     roi = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
#     transformations.append((roi, "ROI objects"))
#
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     analyzed_img = img_rgb.copy()
#     cv2.drawContours(analyzed_img, contours, -1, (0, 255, 0), 2)
#     transformations.append((analyzed_img, "Analyze object"))
#
#     if len(contours) > 0:
#         largest_contour = max(contours, key=cv2.contourArea)
#         landmarks_img = img_rgb.copy()
#         step = max(1, len(largest_contour) // 30)
#         for i in range(0, len(largest_contour), step):
#             x, y = largest_contour[i][0]
#             cv2.circle(landmarks_img, (x, y), 5, (255, 0, 0), -1)
#         transformations.append((landmarks_img, "Pseudolandmarks"))
#     else:
#         transformations.append((img_rgb.copy(), "No contours found for pseudolandmarks"))
#
#     hist_img = create_color_histogram(img_rgb)
#     transformations.append((hist_img, "Color histogram"))
#
#     if destination:
#         filename = os.path.basename(image_path)
#         name, ext = os.path.splitext(filename)
#
#         for i, (img, title) in enumerate(transformations):
#             save_path = os.path.join(destination, f"{name}_{i + 1}_{title.replace(' ', '_')}{ext}")
#             if len(img.shape) == 2:
#                 cv2.imwrite(save_path, img)
#             else:
#                 cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#         print(f"Saved transformations for {filename} to {destination}")
#     else:
#         fig, axes = plt.subplots(2, 4, figsize=(20, 10))
#         axes = axes.flatten()
#
#         for i, (img, title) in enumerate(transformations):
#             if i < len(axes):
#                 if len(img.shape) == 2:
#                     axes[i].imshow(img, cmap='gray')
#                 else:
#                     axes[i].imshow(img)
#                 axes[i].set_title(f"Figure IV.{i + 1}: {title}")
#                 axes[i].axis('off')
#
#         for i in range(len(transformations), len(axes)):
#             axes[i].axis('off')
#
#         plt.tight_layout()
#         plt.show()


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
