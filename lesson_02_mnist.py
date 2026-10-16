import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the dataset.
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# # Inspect the dataset.
# print("TensorFlow version:", tf.__version__)

# print("Training images:", x_train.shape)
# print("Training labels:", y_train.shape)

# print("Test images:", x_test.shape)
# print("Test labels:", y_test.shape)

# print("Image data type:", x_train.dtype)
# print("Minimum pixel value:", x_train.min())
# print("Maximum pixel value:", x_train.max())

# # Display the first 10 training images.
# plt.figure(figsize=(10, 4))

# for i in range(10):
#     plt.subplot(2, 5, i + 1)

#     plt.imshow(
#         x_train[i],
#         cmap="gray",
#         vmin=0,
#         vmax=255
#     )

#     plt.title(f"Label: {y_train[i]}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

# # Inspect one training image and its label.
# sample_image = x_train[0]
# sample_label = y_train[0]

# print("Sample shape:", sample_image.shape)
# print("Correct label:", sample_label)
# print("Data type:", sample_image.dtype)

# # Inspect one pixel.
# print("Pixel at row 14, column 14:", sample_image[14, 14])

# # Print a small 5 × 5 region of the image.
# print("Small pixel region:")
# print(sample_image[10:15, 10:15])

# Convert pixel values from 0–255 to 0–1.
x_train_cnn = x_train.astype(np.float32) / 255.0
x_test_cnn = x_test.astype(np.float32) / 255.0

# print("Original pixel:", x_train[0, 14, 14])
# print("Normalized pixel:", x_train_cnn[0, 14, 14])

# print("Original data type:", x_train.dtype)
# print("Prepared data type:", x_train_cnn.dtype)

# print("Normalized minimum:", x_train_cnn.min())
# print("Normalized maximum:", x_train_cnn.max())

# Add one grayscale channel to each image.
x_train_cnn = x_train_cnn[..., np.newaxis]
x_test_cnn = x_test_cnn[..., np.newaxis]

# print("CNN training images:", x_train_cnn.shape)
# print("CNN test images:", x_test_cnn.shape)

# Labels are still integers from 0 to 9.
print("First 10 labels:", y_train[:10])
print("Available digit classes:", np.unique(y_train))