# #!/usr/bin/env python3
# import cv2
# import numpy as np
# import argparse
# import os
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# from plantcv import plantcv as pcv
#
#
# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Leaf Image Transformation Tool')
#     parser.add_argument('path', nargs='?', help='Path to a single image or source directory')
#     parser.add_argument('-src', '--source', help='Source directory containing images')
#     parser.add_argument('-dst', '--destination', help='Destination directory for saving transformed images')
#     parser.add_argument('-mask', action='store_true', help='Apply masking operations')
#
#     return parser.parse_args()
#
# def create_color_histogram(img_rgb):
#     """Create a color histogram visualization from an RGB image"""
#     # Create a figure for the histogram
#     fig = plt.figure(figsize=(8, 6))
#
#     # Get the histogram data for each channel
#     colors = ('red', 'green', 'blue')
#     color_labels = ['red', 'blue-yellow', 'green', 'green-magenta', 'hue', 'lightness', 'red', 'saturation', 'value']
#     color_values = ['r', 'c', 'g', 'm', 'violet', 'gray', 'r', 'y', 'b']
#
#     # RGB histograms
#     for i, color in enumerate(colors):
#         hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
#         plt.plot(hist, color=color_values[i], alpha=0.7, label=color_labels[i])
#
#     # Convert to HSV for additional histograms
#     img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
#
#     # HSV histograms
#     for i, label in enumerate(['hue', 'saturation', 'value']):
#         if i == 0:  # Hue has range 0-179 in OpenCV
#             hist = cv2.calcHist([img_hsv], [i], None, [180], [0, 180])
#             plt.plot(hist, color=color_values[i + 4], alpha=0.7, label=color_labels[i + 4])
#         else:
#             hist = cv2.calcHist([img_hsv], [i], None, [256], [0, 256])
#             plt.plot(hist, color=color_values[i + 6], alpha=0.7, label=color_labels[i + 6])
#
#     # LAB color space (for additional histograms)
#     img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
#
#     # L (lightness) histogram
#     hist = cv2.calcHist([img_lab], [0], None, [256], [0, 256])
#     plt.plot(hist, color=color_values[5], alpha=0.7, label=color_labels[5])
#
#     # a (green-magenta) histogram
#     hist = cv2.calcHist([img_lab], [1], None, [256], [0, 256])
#     plt.plot(hist, color=color_values[3], alpha=0.7, label=color_labels[3])
#
#     # b (blue-yellow) histogram
#     hist = cv2.calcHist([img_lab], [2], None, [256], [0, 256])
#     plt.plot(hist, color=color_values[1], alpha=0.7, label=color_labels[1])
#
#     plt.xlabel('Pixel intensity')
#     plt.ylabel('Proportion of pixels (%)')
#     plt.title('Color Histogram')
#     plt.xlim([0, 256])
#     plt.legend()
#     plt.grid(alpha=0.3)
#
#     # Convert the plot to an image
#     canvas = plt.get_current_fig_manager().canvas
#     canvas.draw()
#     hist_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
#     hist_img = hist_img.reshape(canvas.get_width_height()[::-1] + (3,))
#
#     plt.close()
#
#     return hist_img
#
#
# def process_image(image_path, destination=None, apply_mask=False):
#     """Process a single image with various transformations using PlantCV"""
#     # Read the image
#     # img = cv2.imread(image_path)
#     img, path, filename = pcv.readimage(image_path)
#     img_rgb = img.copy()
#     if img is None:
#         print(f"Error: Could not read the image: {image_path}")
#         return
#
#     # Create a list to store images and their titles
#     transformations = []
#
#     # 1. Original image
#     # Convert BGR to RGB for display
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     transformations.append((img_rgb, "Original"))
#
#     # 2. Gaussian blur - using PlantCV's wrapper
#     gaussian_blur = pcv.gaussian_blur(img_rgb, ksize=(5, 5))
#     transformations.append((gaussian_blur, "Gaussian blur"))
#
#     img_rgb = gaussian_blur
#
#     # 3. Create mask with PlantCV
#     # Convert to HSV color space
#
#     s = pcv.rgb2gray_hsv(img_rgb, 's')
#     s_thresh = pcv.threshold.binary(s, 80, 'light')
#     s_mblur = pcv.median_blur(s_thresh, 5)
#
#     b = pcv.rgb2gray_lab(img_rgb, 'a')
#     # Threshold the blue image
#     b_thresh = pcv.threshold.binary(b, 130, 'dark')
#     # Fill small objects
#     b_fill = pcv.fill(b_thresh, 5)
#
#     # Combine the thresholds
#     mask = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_fill)
#
#     # # Clean up the mask
#     mask = pcv.fill(mask, 100)  # Fill small holes
#     mask = pcv.dilate(mask, 3, 1)  # Dilate to capture more of the leaf
#     mask = pcv.erode(mask, 3, 1)  # Erode to clean edges
#
#     transformations.append((mask, "mask"))
#
#     # 4. Apply Mask to the image
#     # Apply the mask to the original image
#     mask = pcv.fill(mask, 3)
#     applied = pcv.apply_mask(img=img_rgb, mask=mask, mask_color='white')
#     transformations.append((applied, "Applied Mask"))
#
#     # 5. Analyze image
#     shape_img = pcv.analyze.size(img=img_rgb, labeled_mask=mask, n_labels=1)
#     pcv.outputs.save_results(filename="results.txt", outformat="json")
#     transformations.append((shape_img, "Analyze Object"))
#
#
#     # 6. Pseudolandmarks
#     pseudo_img = img_rgb.copy()
#     pcv.params.debug_outdir = "./temp/"
#     top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(img=pseudo_img, mask=mask, label="default")
#     # Draw top pseudolandmarks in red
#     for point in top_x:
#         x, y = int(point[0][0]), int(point[0][1])
#         cv2.circle(pseudo_img, (x, y), 3, (255, 0, 0), -1)
#     # Draw center pseudolandmarks in green
#     for point in center_v_x:
#         x, y = int(point[0][0]), int(point[0][1])
#         cv2.circle(pseudo_img, (x, y), 3, (0, 255, 0), -1)
#     # Draw bottom pseudolandmarks in blue
#     for point in bottom_x:
#         x, y = int(point[0][0]), int(point[0][1])
#         cv2.circle(pseudo_img, (x, y), 3, (0, 0, 255), -1)
#     transformations.append((pseudo_img, "Pseudolandmarks"))
#
#
#     # 7. Create and add color histogram
#     # Compute the histograms for each channel
#     hist_img = create_color_histogram(img_rgb)
#     transformations.append((hist_img, "Color Histogram"))
#
#     # Display or save the transformations
#     if destination:
#         # Save all transformations to the destination folder
#         base_name = os.path.basename(image_path)
#         file_name, ext = os.path.splitext(base_name)
#
#         for img_trans, title in transformations:
#             save_path = os.path.join(destination, f"{file_name}_{title.lower().replace(' ', '_')}{ext}")
#             # Convert back to BGR for saving with OpenCV
#             if len(img_trans.shape) == 3 and img_trans.shape[2] == 3:
#                 img_save = cv2.cvtColor(img_trans, cv2.COLOR_RGB2BGR)
#             else:
#                 img_save = img_trans
#             cv2.imwrite(save_path, img_save)
#     else:
#         # Display all transformations
#         fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#         axes = axes.flatten()
#
#         for i, (img_trans, title) in enumerate(transformations):
#             if i < len(axes):
#                 if len(img_trans.shape) == 2:  # Grayscale image
#                     axes[i].imshow(img_trans, cmap='gray')
#                 else:  # Color image
#                     axes[i].imshow(img_trans)
#                 axes[i].set_title(title)
#                 axes[i].axis('off')
#
#         for ax in axes[7:]:
#             ax.axis('off')
#
#         plt.tight_layout()
#         plt.show()
#
#     return transformations
#
#
# def main():
#     args = parse_arguments()
#
#     # Create destination directory if specified and doesn't exist
#     if args.destination and not os.path.exists(args.destination):
#         os.makedirs(args.destination)
#
#     if args.path and os.path.isfile(args.path):
#         process_image(args.path, args.destination, args.mask)
#
#     elif args.source and os.path.isdir(args.source):
#         if not args.destination:
#             print("Error: Destination directory (-dst) is required when processing a directory")
#             return
#
#         for filename in os.listdir(args.source):
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
#                 image_path = os.path.join(args.source, filename)
#                 process_image(image_path, args.destination, args.mask)
#
#     else:
#         print("Error: Please provide a valid image path or source directory")
#
#
# if __name__ == "__main__":
#     main()
# !/usr/bin/env python3
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
    parser.add_argument('path', nargs='?', help='Path to a single image to display transformations')
    parser.add_argument('-src', '--source', help='Source directory containing images to process and save')
    parser.add_argument('-dst', '--destination', help='Destination directory for saving transformed images')

    # Add transformation options as boolean flags
    parser.add_argument('-original', action='store_true', help='Save original image')
    parser.add_argument('-blur', action='store_true', help='Apply Gaussian blur transformation')
    parser.add_argument('-mask', action='store_true', help='Generate binary mask')
    parser.add_argument('-masked', action='store_true', help='Apply mask to the original image')
    parser.add_argument('-analyze', action='store_true', help='Analyze object size and shape')
    parser.add_argument('-landmarks', action='store_true', help='Generate pseudolandmarks')
    parser.add_argument('-histogram', action='store_true', help='Generate color histogram')
    parser.add_argument('-all', action='store_true', help='Apply all transformations')

    return parser.parse_args()


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
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    hist_img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    hist_img = cv2.cvtColor(hist_img, cv2.COLOR_RGBA2RGB)

    plt.close()

    return hist_img


def load_image(image_path):
    """Load an image using PlantCV"""
    img, path, filename = pcv.readimage(image_path)
    if img is None:
        print(f"Error: Could not read the image: {image_path}")
        return None, None, None

    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, path, filename


def apply_gaussian_blur(img_rgb):
    """Apply Gaussian blur to the image"""
    return pcv.gaussian_blur(img_rgb, ksize=(5, 5))


def create_mask(img_rgb):
    """Create a binary mask for the plant"""
    # Convert to HSV color space for saturation channel
    s = pcv.rgb2gray_hsv(img_rgb, 's')
    s_thresh = pcv.threshold.binary(s, 80, 'light')
    s_mblur = pcv.median_blur(s_thresh, 5)

    # Convert to LAB color space for a channel
    b = pcv.rgb2gray_lab(img_rgb, 'a')
    # Threshold the blue image
    b_thresh = pcv.threshold.binary(b, 130, 'dark')
    # Fill small objects
    b_fill = pcv.fill(b_thresh, 5)

    # Combine the thresholds
    mask = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_fill)

    # Clean up the mask
    mask = pcv.fill(mask, 100)  # Fill small holes
    mask = pcv.dilate(mask, 3, 1)  # Dilate to capture more of the leaf
    mask = pcv.erode(mask, 3, 1)  # Erode to clean edges
    mask = pcv.fill(mask, 3)  # Final fill

    return mask


def apply_mask_to_image(img_rgb, mask):
    """Apply the binary mask to the original image"""
    return pcv.apply_mask(img=img_rgb, mask=mask, mask_color='white')


def analyze_object(img_rgb, mask):
    """Analyze the object size and shape"""
    shape_img = pcv.analyze.size(img=img_rgb, labeled_mask=mask, n_labels=1)
    pcv.outputs.save_results(filename="results.txt", outformat="json")
    return shape_img


def generate_pseudolandmarks(img_rgb, mask):
    """Generate pseudolandmarks for the object"""
    pseudo_img = img_rgb.copy()
    pcv.params.debug_outdir = "./temp/"
    top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(img=pseudo_img, mask=mask, label="default")

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

    return pseudo_img


def save_image(img, save_path):
    """Save an image to the specified path"""
    # Convert back to BGR for saving with OpenCV if it's an RGB image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_save = img
    cv2.imwrite(save_path, img_save)


def display_images(transformations):
    """Display all transformations in a grid"""
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


def get_selected_transformations(args):
    """Determine which transformations to apply based on args"""
    # If no specific transformations are selected but -all is not used,
    # apply all transformations by default
    if not any([args.original, args.blur, args.mask, args.masked,
                args.analyze, args.landmarks, args.histogram, args.all]):
        return True  # Apply all transformations

    # Otherwise, return the -all flag
    return args.all


def process_image(image_path, destination=None, args=None, display=False):
    """Process a single image with transformations"""
    # Load the image
    img_rgb, path, filename = load_image(image_path)
    if img_rgb is None:
        return

    # Create a list to store transformations that are actually performed
    transformations = []
    base_name = os.path.basename(image_path)
    file_name, ext = os.path.splitext(base_name)

    # Check which transformations to apply based on arguments
    apply_all = get_selected_transformations(args)

    # 1. Original image
    if args.original or apply_all:
        transformations.append((img_rgb, "Original"))
        if destination:
            save_path = os.path.join(destination, f"{file_name}_original{ext}")
            save_image(img_rgb, save_path)

    # 2. Gaussian blur
    if args.blur or apply_all:
        gaussian_blur = apply_gaussian_blur(img_rgb)
        transformations.append((gaussian_blur, "Gaussian blur"))
        if destination:
            save_path = os.path.join(destination, f"{file_name}_gaussian_blur{ext}")
            save_image(gaussian_blur, save_path)

        # Use blurred image for subsequent operations
        img_rgb = gaussian_blur

    # 3. Create mask
    mask = None
    if args.mask or args.masked or args.analyze or args.landmarks or apply_all:
        mask = create_mask(img_rgb)
        if args.mask or apply_all:
            transformations.append((mask, "Mask"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_mask{ext}")
                save_image(mask, save_path)

    # 4. Apply mask to image
    if args.masked or apply_all:
        if mask is not None:
            applied = apply_mask_to_image(img_rgb, mask)
            transformations.append((applied, "Applied Mask"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_applied_mask{ext}")
                save_image(applied, save_path)

    # 5. Analyze object
    if args.analyze or apply_all:
        if mask is not None:
            shape_img = analyze_object(img_rgb, mask)
            transformations.append((shape_img, "Analyze Object"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_analyze_object{ext}")
                save_image(shape_img, save_path)

    # 6. Generate pseudolandmarks
    if args.landmarks or apply_all:
        if mask is not None:
            pseudo_img = generate_pseudolandmarks(img_rgb, mask)
            transformations.append((pseudo_img, "Pseudolandmarks"))
            if destination:
                save_path = os.path.join(destination, f"{file_name}_pseudolandmarks{ext}")
                save_image(pseudo_img, save_path)

    # 7. Generate color histogram
    if args.histogram or apply_all:
        hist_img = create_color_histogram(img_rgb)
        transformations.append((hist_img, "Color Histogram"))
        if destination:
            save_path = os.path.join(destination, f"{file_name}_color_histogram{ext}")
            save_image(hist_img, save_path)

    # Display transformations if requested
    if display and transformations:
        display_images(transformations)

    return transformations


def main():
    args = parse_arguments()

    # Create destination directory if specified and doesn't exist
    if args.destination and not os.path.exists(args.destination):
        os.makedirs(args.destination)

    # CASE 1: Direct path to an image - display transformations
    if args.path and os.path.isfile(args.path):
        print(f"Processing single image: {args.path}")
        process_image(args.path, destination=args.destination, args=args, display=True)

    # CASE 2: Source directory provided - save transformations to destination
    elif args.source and os.path.isdir(args.source):
        if not args.destination:
            print("Error: Destination directory (-dst) is required when processing a directory")
            return

        print(f"Processing all images in directory: {args.source}")
        print(f"Saving transformations to: {args.destination}")

        for filename in os.listdir(args.source):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(args.source, filename)
                process_image(image_path, destination=args.destination, args=args, display=False)

    # No valid input
    else:
        print("Error: Please provide a valid image path or source directory")
        parser = argparse.ArgumentParser()
        parser.print_help()


if __name__ == "__main__":
    main()
