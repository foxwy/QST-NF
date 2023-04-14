# Discrete Denoising Flows

This repository contains the code for the experiments presented in the paper [_Discrete Denoising
Flows_](https://arxiv.org/abs/2107.11625) [1].

To give a short overview on the architecture of the implementation:

- `main.py`: Starting point and configuration of experiments
- `training.py`: Training logic
- `visualization_.py`: Functions for plotting samples from trained model
- `model/categorical_prior.py`: Prior distribution and splitpriors
- `model/model.py`: Overall model object (Discrete Denoising Flow and prior)
- `model/flow.py`:  Discrete Denoising Flow object
- `model/flow_layers.py`: Implementations of
    - **Discrete denoising coupling layer (including the conditional permutation operation introduced in the paper)**
    - Permutation layer
    - Squeeze layer
- `model/network.py`: Implementation of DenseNet and simple MLP
- `data/*`: Logic for loading _Eight Gaussians_, _MNIST_ and _Cityscapes_ datasets

## Usage

For each of the following commands, the results are saved in the folder `./results`.

### 8 Gaussians

To test _Discrete Denoising Flows_ with limited computational resources, run the 8 Gaussian toy data experiment. It
takes only a few minutes to execute on a 12 GB RAM laptop.

```
python main.py --dataset='8gaussians' --k_sort=91 --n_hidden_nn=256 --net_epochs=30 --prior_epochs=20
```

### Binary MNIST

For the experiment on Binary MNIST run

```
python main.py --dataset='mnist' --k_sort=2 --n_hidden_nn=512 --densenet_depth=10 --net_epochs=100 --prior_epochs=30 
```

For running the experiment without splitpriors, set the flag `--with_splitprior False`.

### Cityscapes

For this experiment, it is necessary to download the [Cityscapes data set](https://www.cityscapes-dataset.com). 
For preprocessing, download from 
[this repository](https://github.com/ehoogeboom/multinomial_diffusion/tree/main/segmentation_diffusion/cityscapes)
the `data_to_npy.py` and `cityscapes.py` files that perform the
conversion of the original data. This creates three `.npy` files that should be placed
in `./data/cityscapes/preprocessed`.
Then run 
```
python main.py --dataset='cityscapes' --k_sort=4 --n_hidden_nn=512 --densenet_depth=15 --net_epochs=100 --prior_epochs=30 
```

Again, for running the experiment without splitpriors, set the flag `--with_splitprior False`.

# Acknowledgements

We gratefully acknowledge the financial support of Robert Bosch GmbH.

# References

[1] Alexandra Lindt and Emiel Hoogeboom. "Discrete Denoising Flows." ICML Workshop on Invertible Neural Networks,
Normalizing Flows, and Explicit Likelihood Models (2021).
