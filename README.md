# DigitVision — Handwritten Digit Recognition with CNNs

DigitVision is a handwritten digit recognition project built with Python and TensorFlow/Keras. It uses a Convolutional Neural Network (CNN) to classify grayscale images of handwritten digits from 0 to 9.

The project focuses on understanding the complete CNN workflow, from image preprocessing and convolution to model training, prediction, and evaluation.

## Features

- Handwritten digit classification using the MNIST dataset
- CNN architecture implemented with TensorFlow/Keras
- Image normalization and channel-dimension preprocessing
- Standalone prediction script
- Model evaluation on the MNIST test dataset
- Per-digit precision and recall
- Confusion matrix visualization saved as an image

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- MNIST dataset

## CNN Architecture

The model processes 28 × 28 grayscale images.

| Layer | Configuration | Output shape |
|---|---|---|
| Input | Grayscale image | 28 × 28 × 1 |
| Conv2D | 32 filters, 3 × 3, ReLU, same padding | 28 × 28 × 32 |
| MaxPooling2D | 2 × 2 | 14 × 14 × 32 |
| Conv2D | 64 filters, 3 × 3, ReLU, same padding | 14 × 14 × 64 |
| MaxPooling2D | 2 × 2 | 7 × 7 × 64 |
| Flatten | Convert feature maps to a vector | 3,136 |
| Dense | 64 units, ReLU | 64 |
| Output | 10 units, softmax | 10 probabilities |

The convolutional layers learn visual features, pooling reduces spatial dimensions, and the dense layers use those features to classify the digit.

## Dataset and Preprocessing

DigitVision uses MNIST, a dataset of 70,000 grayscale handwritten digit images:

- 60,000 training images
- 10,000 test images
- Image dimensions: 28 × 28 pixels
- Number of classes: 10

Pixel values are normalized from the range 0–255 to 0–1. A channel dimension is added so each image has the shape expected by the CNN: `(28, 28, 1)`.

The test dataset is used for final evaluation, not for fitting model parameters.

## Project Structure

```text
DigitVision/
├── models/
│   └── digitvision_cnn.keras
├── src/
│   ├── predict.py
│   └── evaluate.py
├── reports/
│   └── confusion_matrix.png
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

The notebook directory contains the development notebook. The saved model is loaded by the standalone scripts.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DigitVision
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Use a Python version compatible with the TensorFlow version specified in your environment.

## Run a Prediction

From the project root:

```powershell
python src\predict.py
```

To predict a different MNIST test image:

```powershell
python src\predict.py --index 25
```

The script displays the image, actual label, predicted digit, and model confidence.

## Evaluate the Model

Run:

```powershell
python src\evaluate.py
```

The evaluation script prints test accuracy, correct and incorrect prediction counts, per-digit precision and recall, and the most frequent confusion.

It also generates:

`reports/confusion_matrix.png`

## Results

The evaluation script calculates results directly from the saved model and MNIST test dataset. Run it in your environment to obtain the current metrics.

## Learning Outcomes

Through this project, I explored:

- Image representation and pixel normalization
- Convolution, filters, feature maps, and stride
- ReLU activation and max pooling
- CNN architecture and output shapes
- Softmax probabilities and classification
- Loss functions, backpropagation, and optimization
- Model evaluation, precision, recall, and confusion matrices
- Saving and loading trained models for inference

## Future Improvements

- Interactive handwritten digit drawing and prediction interface
- Visualization of learned convolutional feature maps
- Analysis of incorrectly classified images
- Comparison of CNN architectures and hyperparameters

## License

Add a license if you intend to distribute this project publicly.