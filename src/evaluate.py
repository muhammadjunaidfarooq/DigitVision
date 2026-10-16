
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Define project paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "digitvision_cnn.keras"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Check that the model exists inside the models folder."
        )

    # Load the trained model.
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load the independent MNIST test dataset.
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Apply the same preprocessing used during training.
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., np.newaxis]

    # Predict all test images.
    probabilities = model.predict(x_test, batch_size=128, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    # Calculate overall accuracy.
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print(f"Correct predictions: {np.sum(y_pred == y_test)}")
    print(f"Incorrect predictions: {np.sum(y_pred != y_test)}")

    # Create the confusion matrix.
    cm = tf.math.confusion_matrix(
        y_test,
        y_pred,
        num_classes=10,
    ).numpy()

    print("\nConfusion matrix:")
    print(cm)

    # Calculate precision and recall for each digit.
    print("\nPer-digit results:")
    print("Digit | Precision | Recall")

    for digit in range(10):
        true_positive = cm[digit, digit]
        false_positive = cm[:, digit].sum() - true_positive
        false_negative = cm[digit, :].sum() - true_positive

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator else 0.0
        )

        print(
            f"  {digit}   |  {precision * 100:6.2f}%  |"
            f" {recall * 100:6.2f}%"
        )

    # Find the most frequent incorrect actual -> predicted pair.
    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    if errors.max() > 0:
        actual_digit, predicted_digit = np.unravel_index(
            np.argmax(errors), errors.shape
        )
        count = errors[actual_digit, predicted_digit]

        print(
            f"\nMost frequent confusion: actual {actual_digit} "
            f"predicted as {predicted_digit} ({count} images)"
        )
    else:
        print("\nNo incorrect predictions found.")

    # Plot and save the confusion matrix.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of images")

    ax.set_title("DigitVision CNN — Confusion Matrix")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("Actual digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    # Annotate every cell with its count.
    threshold = cm.max() / 2
    for row in range(10):
        for col in range(10):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()

    report_path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(report_path, dpi=200, bbox_inches="tight")
    print(f"\nConfusion matrix saved to: {report_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
