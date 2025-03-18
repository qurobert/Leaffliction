import argparse
import numpy as np
import keras
import json
import os

from tensorflow import truediv


def predict_image(path, model, class_names):
    img = keras.utils.load_img(path, target_size=(256, 256))
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    prediction = class_names[predicted_class]
    print(f"Predicted class for {path}: {prediction}")
    return prediction.lower() in path.lower()




def argument_parser():
    parser = argparse.ArgumentParser(
        prog="Leaf computer vision to classify plant disease",
        description="This program balanced the dataset, transform it and train with scikit-learn")
    parser.add_argument('path',
                        type=str,
                        help="Path to an image file or a directory containing images to predict")
    parser.add_argument('--model_filename',
                        type=str,
                        default="data/model/model.keras",
                        help="The model file")
    return parser.parse_args()


def main():
    args = argument_parser()
    path, model_filename = args.path, args.model_filename
    model = keras.models.load_model(model_filename)
    with open("./data/model/class_names.json", "r") as f:
        class_names = json.load(f)

    if os.path.isfile(path):
        result = predict_image(path, model, class_names)
        if result:
            print("Good prediction")
        else:
            print("Bad prediction")
    elif os.path.isdir(path):
        result_count = 0
        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                    file_path = os.path.join(root, filename)
                    result = predict_image(file_path, model, class_names)
                    if result:
                        result_count += 1
                    total += 1
                else:
                    print(f"Skipping {filename} (not an image)")
        print(f"Accuracy: {truediv(result_count, total)}")
    else:
        raise ValueError("Path is not a file or directory")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

