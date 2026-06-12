"""
Inference helper for the Food-101 transfer-learning model.

Loads a trained EfficientNet model (Keras SavedModel format) and classifies a
single image, printing the predicted class and confidence.

Usage:
    python src/predict.py --model saved_model/ --image path/to/food.jpg
"""
import argparse

import numpy as np
import tensorflow as tf

# The 10-class Food-101 subset used in the notebook, in label-index order.
CLASS_NAMES = [
    "chicken_curry",
    "chicken_wings",
    "fried_rice",
    "grilled_salmon",
    "hamburger",
    "ice_cream",
    "pizza",
    "ramen",
    "steak",
    "sushi",
]

IMG_SIZE = (224, 224)


def load_and_prep_image(path: str) -> tf.Tensor:
    """Read an image from disk and resize it to the model's input shape.

    EfficientNet in tf.keras expects raw 0-255 pixel values (it rescales
    internally), so we deliberately do NOT divide by 255 here.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.expand_dims(img, axis=0)  # add the batch dimension


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a food image.")
    parser.add_argument("--model", required=True, help="Path to the saved Keras model.")
    parser.add_argument("--image", required=True, help="Path to the image to classify.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    batch = load_and_prep_image(args.image)

    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"

    print(f"Prediction: {label}")
    print(f"Confidence: {probs[idx]:.2%}")


if __name__ == "__main__":
    main()
