# 02. Data pipeline

The dataset comes as two CSV files from Kaggle. Each row has one label and 784 pixel values. The first coding step was to separate the label column from the pixel columns.

The pixel values were then reshaped into `(28, 28, 1)`. I kept the final channel dimension even though the images are grayscale, because TensorFlow convolution layers expect image tensors with a channel axis.

I normalised the pixels from the 0-255 range to the 0-1 range. This makes training easier because the optimiser does not have to work with large raw pixel values.

The labels needed one extra step. Sign Language MNIST uses alphabet indices from 0 to 24, but J is missing because it is a motion-based sign. I remapped the labels into continuous class indices from 0 to 23. That made the labels compatible with sparse categorical cross-entropy and a 24-unit softmax output layer.

For validation, I used a stratified 85/15 split of the training CSV. The official test CSV was kept untouched until final evaluation.
