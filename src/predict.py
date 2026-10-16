
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Find the project root, regardless of where the script is launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "digitvision_cnn.keras"


def main():
    parser = argparse.ArgumentParser(
        description="Predict a handwritten digit using DigitVision."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the MNIST test image (0 to 9999).",
    )
    args = parser.parse_args()

    # Validate the requested image index.
    if not 0 <= args.index < 10000:
        raise ValueError("Index must be between 0 and 9999.")

    # Load the saved CNN.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Check that digitvision_cnn.keras is inside the models folder."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # Load the MNIST test dataset.
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Select one image and prepare it for the CNN.
    image = x_test[args.index]
    image_input = image.astype("float32") / 255.0
    image_input = image_input[np.newaxis, ..., np.newaxis]

    # Predict the digit.
    probabilities = model.predict(image_input, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100
    actual_digit = int(y_test[args.index])

    # Display the result.
    print(f"Image index: {args.index}")
    print(f"Actual digit: {actual_digit}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Prediction confidence: {confidence:.2f}%")

    plt.imshow(image, cmap="gray")
    plt.title(
        f"Actual: {actual_digit} | Predicted: {predicted_digit}\n"
        f"Confidence: {confidence:.2f}%"
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
