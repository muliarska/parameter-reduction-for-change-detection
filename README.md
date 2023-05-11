# Comparison of Parameter reduction methods for Change Detection in Satellite Imagery

The GitHub repository contains the implementation of Yana Muliarska's thesis, "Comparison of Parameter reduction methods for Change Detection in Satellite Imagery", supervised by Petr Simanek.

## Structure 

The [parameters_reduction.py](https://github.com/muliarska/parameter-reduction-for-change-detection/blob/main/algorithms/parameters_reduction.py) file in the 'algorithms' folder contains the implementations of parameter reduction algorithms: Fractional Filter, Pruning, and Low-Rank Approximation.

The 'experiments' folder contains our experiments on CNN trained on the MNIST dataset:
- [1conv_mnist_params_reduction.ipynb](https://github.com/muliarska/parameter-reduction-for-change-detection/blob/main/experiments/MNIST_CNN/1conv_mnist_params_reduction.ipynb): general experimental setup.
- [2conv_mnist_params_reduction.ipynb](https://github.com/muliarska/parameter-reduction-for-change-detection/blob/main/experiments/MNIST_CNN/2conv_mnist_params_reduction.ipynb): application of methods to different convolutional layers.
- [kernels_mnist_params_reduction.ipynb](https://github.com/muliarska/parameter-reduction-for-change-detection/blob/main/experiments/MNIST_CNN/kernels_mnist_params_reduction.ipynb): kernel size modification.

And the experiments on the change detection model SNUNet-CD:
- [run_SNUNet_CD_original.ipynb](https://github.com/muliarska/parameter-reduction-for-change-detection/blob/main/experiments/SNUNet-CD/run_SNUNet_CD_original.ipynb): original model architecture.
- [run_SNUNet_CD_fractional.ipynb](https://github.com/muliarska/parameter-reduction-for-change-detection/blob/main/experiments/SNUNet-CD/run_SNUNet_CD_fractional.ipynb): Fractional Filter.
- [run_SNUNet_CD_pruning.ipynb](https://github.com/muliarska/parameter-reduction-for-change-detection/blob/main/experiments/SNUNet-CD/run_SNUNet_CD_pruning.ipynb): Pruning.
- [run_SNUNet_CD_lowrank.ipynb](https://github.com/muliarska/parameter-reduction-for-change-detection/blob/main/experiments/SNUNet-CD/run_SNUNet_CD_lowrank.ipynb): Low-Rank Approximation.
