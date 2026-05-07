# 04. Enhanced VGG-style model

After the baseline, I built a deeper VGG-style CNN. This model uses three convolutional blocks. Each block has two 3 by 3 convolution layers, batch normalisation, max pooling, and dropout.

The filter counts increase from 32 to 64 to 128. The idea is that early layers learn simple edge patterns, while later layers learn more specific hand-shape features.

I also used L2 regularisation and global average pooling. L2 regularisation discourages very large weights. Global average pooling replaces a large flattening layer, so the model has fewer dense parameters.

This was useful because the enhanced model is not just a larger version of the baseline. It actually has fewer parameters than the baseline, but it uses them in a deeper convolutional structure.
