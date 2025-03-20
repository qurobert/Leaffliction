import argparse

import cv2
import numpy as np
import keras
import json
import os
from tensorflow import truediv
import matplotlib.pyplot as plt
from Transformation import process_image
from train_preprocessing import remove_background_cv2_part


def predict_image(path, model, class_names, is_display=True):
    img = keras.utils.load_img(path, target_size=(256, 256))
    results = process_image(path, None, ["masked"], False)
    # img_back = remove_background_cv2_part(path, None, None)

    img_array = keras.utils.img_to_array(img)
    img_array = keras.utils.img_to_array(results["masked"])
    img_array = keras.ops.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    prediction = class_names[predicted_class]
    result = prediction.lower() in path.lower()
    if not result:
        print("RESULT FALSE FOR : ", prediction.lower(), path.lower())
    print(f"Predicted class for {path}: {prediction}")
    if is_display:
        display_image(img,
                      results["masked"],
                      prediction,
                      os.path.splitext(os.path.basename(path))[0],
                      result)

    return result


def display_image(img, img_without_background, prediction, original_name, result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img)
    axes[0].axis("off")

    img_without_background_rgb = cv2.cvtColor(img_without_background, cv2.COLOR_BGR2RGB)
    axes[1].imshow(img_without_background_rgb)
    axes[1].axis("off")

    plt.figtext(0.5, 0.08,
                "=== Dl classification ===",
                ha="center",
                fontsize=16)
    if not result:
        plt.figtext(0.5, 0.02,
                    f"{prediction} (≠ {original_name})",
                    ha="center",
                    fontsize=12,
                    color="red")
    else:
        fig.text(0.5, 0.02,
                 f"{prediction}",
                 ha="center",
                 fontsize=12,
                 color="green")
    plt.show()


def argument_parser():
    parser = argparse.ArgumentParser(
        prog="Leaf computer vision to classify plant disease",
        description="This program predict from training with CNN model,"
                    " and can take a filename as argument"
                    " or directory to test the accuracy")
    parser.add_argument('path',
                        type=str,
                        help="Path to an image file"
                             " or a directory containing images to predict")
    parser.add_argument('--model_dir',
                        type=str,
                        default="data/model",
                        help="The model file")
    return parser.parse_args()


def main():
    args = argument_parser()
    path, model_dir = args.path, args.model_dir
    model = keras.models.load_model(os.path.join(model_dir, "model.keras"))
    with open(os.path.join(model_dir, "class_names.json"), "r") as f:
        class_names = json.load(f)

    if os.path.isfile(path):
        result = predict_image(path, model, class_names)
        print("Good prediction !" if result else "Bad prediction...")
    elif os.path.isdir(path):
        result_count = 0
        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                if (filename.lower().endswith(".jpg")
                        or filename.lower().endswith(".png")):
                    file_path = os.path.join(root, filename)
                    result = predict_image(file_path,
                                           model,
                                           class_names,
                                           is_display=False)
                    if result:
                        result_count += 1
                    total += 1
                else:
                    print(f"Skipping {filename} (not an image)")
        print(f"Accuracy: {truediv(result_count, total)}")
    else:
        raise ValueError("Path is not a file or directory")


if __name__ == "__main__":
    # try:
    main()
    # except Exception as e:
    #     print(e)
