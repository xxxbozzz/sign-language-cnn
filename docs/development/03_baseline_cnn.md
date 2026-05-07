# 03. Baseline CNN

I wrote the baseline model first because I needed a simple reference point. The baseline uses two convolution layers, ReLU activation, He normal initialisation, max pooling, a dense layer, dropout, and a softmax classifier.

The model is intentionally simple. If the baseline already worked very well on the official test set, then a deeper model would be less necessary. If it showed a validation-test gap, then the enhanced model would have a clear reason to exist.

The baseline also helped check the data pipeline. When a simple CNN can train and produce sensible results, it is easier to trust that the labels, image shapes, and train/validation split are correct.
