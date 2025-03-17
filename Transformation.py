#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Leaf Image Transformation Tool')
    parser.add_argument('path', nargs='?', help='Path to a single image or source directory')
    parser.add_argument('-src', '--source', help='Source directory containing images')
    parser.add_argument('-dst', '--destination', help='Destination directory for saving transformed images')
    parser.add_argument('-mask', action='store_true', help='Apply masking operations')
    # parser.add_argument('-h', '--help', action='help', help='Show this help message and exit')

    return parser.parse_args()


# Enhanced color histogram function
def create_color_histogram(img_rgb):
    # Create figure with proper size and gray background
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)

    # Calculate and plot RGB histograms
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        hist = hist / img_rgb.size * 100  # Convert to percentage
        plt.plot(hist, color=color, label=color)

    # Convert to HSV for additional histograms
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Hue histogram
    hue_hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
    hue_hist = hue_hist / img_rgb.size * 100  # Convert to percentage
    # Scale to 0-255 for consistent x-axis
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

    # Additional color combinations
    # Green-Magenta channel (from LAB b channel)
    gm_hist = cv2.calcHist([img_lab], [2], None, [256], [0, 256])
    gm_hist = gm_hist / img_rgb.size * 100  # Convert to percentage
    plt.plot(gm_hist, color='magenta', label='green-magenta')

    # Blue-Yellow channel (from LAB a channel)
    by_hist = cv2.calcHist([img_lab], [1], None, [256], [0, 256])
    by_hist = by_hist / img_rgb.size * 100  # Convert to percentage
    plt.plot(by_hist, color='yellow', label='blue-yellow')

    # Add labels and legend
    plt.xlabel('Pixel intensity')
    plt.ylabel('Proportion of pixels (%)')
    plt.legend(title='color Channel', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.ylim(0, 10)  # Set y-axis limit to 10%
    plt.title("Figure IV.7: Color histogram")

    # Save the plot to a numpy array
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    hist_img = np.array(fig.canvas.renderer.buffer_rgba())

    # Close the figure to prevent display
    plt.close()

    return hist_img


def process_image(image_path, destination=None, apply_mask=False):
    """Process a single image with various transformations"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image: {image_path}")
        return

    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a list to store images and their titles
    transformations = []

    # 1. Original image
    transformations.append((img_rgb, "Original"))

    # 2. Gaussian blur
    gaussian_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    transformations.append((gaussian_blur, "Gaussian blur"))

    # 3. Create mask (threshold on green channel)
    # Convert to HSV for better segmentation
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Extract the Saturation channel
    s = hsv[:, :, 1]
    # Threshold the saturation channel
    _, mask = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)
    transformations.append((mask, "Mask"))

    # 4. ROI objects (apply mask to the original image)
    roi = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    transformations.append((roi, "ROI objects"))

    # 5. Analyze object (find contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    analyzed_img = img_rgb.copy()
    cv2.drawContours(analyzed_img, contours, -1, (0, 255, 0), 2)
    transformations.append((analyzed_img, "Analyze object"))

    # 6. Generate pseudolandmarks
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        landmarks_img = img_rgb.copy()
        # Simplified pseudolandmarks (30 points along the contour)
        step = max(1, len(largest_contour) // 30)
        for i in range(0, len(largest_contour), step):
            x, y = largest_contour[i][0]
            cv2.circle(landmarks_img, (x, y), 5, (255, 0, 0), -1)
        transformations.append((landmarks_img, "Pseudolandmarks"))
    else:
        transformations.append((img_rgb.copy(), "No contours found for pseudolandmarks"))

    # 7. Color histogram
    hist_img = create_color_histogram(img_rgb)
    transformations.append((hist_img, "Color histogram"))

    # Display or save the transformations
    if destination:
        # Save the transformations to the destination directory
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        for i, (img, title) in enumerate(transformations):
            save_path = os.path.join(destination, f"{name}_{i + 1}_{title.replace(' ', '_')}{ext}")
            if len(img.shape) == 2:  # If grayscale
                cv2.imwrite(save_path, img)
            else:  # If RGB
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved transformations for {filename} to {destination}")
    else:
        # Display the transformations
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, (img, title) in enumerate(transformations):
            if i < len(axes):
                if len(img.shape) == 2:  # If grayscale
                    axes[i].imshow(img, cmap='gray')
                else:  # If RGB
                    axes[i].imshow(img)
                axes[i].set_title(f"Figure IV.{i + 1}: {title}")
                axes[i].axis('off')

        # Hide any unused subplots
        for i in range(len(transformations), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    args = parse_arguments()

    # Create destination directory if specified and doesn't exist
    if args.destination and not os.path.exists(args.destination):
        os.makedirs(args.destination)

    # Process a single image
    if args.path and os.path.isfile(args.path):
        process_image(args.path, args.destination, args.mask)

    # Process a directory of images
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