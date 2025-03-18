import argparse
import numpy as np
import keras
import json


def predict_image(path, model):
    img = keras.utils.load_img(path, target_size=(256,256))
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    with open("./data/model/class_names.json", "r") as f:
        class_names = json.load(f)
    print(f"Predicted class: {class_names[predicted_class]}")


def argument_parser():
    parser = argparse.ArgumentParser(
        prog="Leaf computer vision to classify plant disease",
        description="This program balanced the dataset, transform it and train with scikit-learn")
    parser.add_argument('filename',
                        type=str,
                        help="Image to predict")
    parser.add_argument('--model_filename',
                        type=str,
                        default="data/model/model.keras",
                        help="The model file")
    return parser.parse_args()


def main():
    args = argument_parser()
    filename, model_filename = args.filename, args.model_filename
    model = keras.models.load_model(model_filename)
    predict_image(filename, model)


if __name__ == "__main__":
    main()
