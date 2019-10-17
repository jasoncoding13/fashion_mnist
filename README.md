# Fashion-MNIST Classification from scratch

## Background
This is a repository to reproduce SVM and CNN from scratch in Numpy:

1. A class of SVM with kernel method and SMO optimization in `./fashion_mnist/svm.py`.
2. A class of 2-(CONV+POOL)+1FC CNN model with SGD, `ConvNet` can be found in `./fashion_mnist/conv_net.py`. In details, there are 3 classes for layers:
    *    `ConvLayer` in `./fashion_mnist/conv_layer.py`
    *   `MaxPoolLayer` in `./fashion_mnist/max_pool_layer.py`
    *    `DenseLayer` in `./fashion_mnist/dense_layer.py`

Note that there are 3 base classes for you to understand the machinism clearly. The actual classes I uesd perform the same behaviour as them with speeding up in vectorized operation.

## Requirement

numpy-1.27.2

## Usage

To be updated...

