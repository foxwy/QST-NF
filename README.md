# **Flow-based Discrete Quantum Distributions Reconstruction**

The official PyTorch implementation of the paper named `Flow-based Discrete Quantum Distributions Reconstruction`.

### **Abstract**

Reconfiguration of the states of multi-body quantum systems is a fundamental and important problem, but is limited by the resource consumption associated with the exponential scaling of the system size, leading to a density matrix reconstruction of generic large quantum systems (>50 qubits) that is essentially impossible to accomplish. Increasingly, generative models approximate quantum states by learning the quantum distribution of quantum states from a finite sample of measurements. The generative models possess linear resource consumption and become a concern by overcoming the curse of dimensionality. Recently, normalizing flow models with explicit probability estimation and fast sampling have attracted much attention in image density estimation, text lossless compression. Here, we apply normalizing flows to the quantum domain for reconstructing discrete quantum distributions. Numerical experiments on quantum systems of various scales show that the normalized flow model maintains similar performance as the autoregressive models while exhibiting excellent scaling of sampling efficiency relative to system size, overcoming the slow sequential sampling of autoregression.

## Getting started

This code was tested on the computer with a single Intel(R) Core(TM) i7-12700KF CPU @ 3.60GHz with 64GB RAM, and a single NVIDIA GeForce RTX 3090 Ti GPU with 24GB RAM, and requires:

- Python 3.7
- conda3
- layers==0.1.5
- matplotlib==3.5.2
- numpy==1.21.6
- openpyxl==3.0.10
- pandas==1.4.4
- prettytable==3.7.0
- scikit_learn==1.2.2
- scipy==1.7.3
- seaborn==0.11.2
- survae==0.1
- tensorboardX==2.6
- tensorflow==2.6.0
- tensorflow_datasets==4.9.2
- torch==1.7.1+cu110
- tqdm==4.64.1
- wandb==0.14.2


You can install code environments by:

```bash
pip install -r requirements.txt
```

## Measured Quantum Samples

```bash
cd "datasets"

# p=1
python data_generation.py --p 1

# p=0.5
python data_generation.py --p 0.5
```

Samples are saved in [`datasets/data`](datasets/data).
If memory error is reported, please set virtual memory.

## Discrete Quantum Distribution Reconstruction with NFs

### RNN model

```bash
cd "models/RNN"

python rnn.py
```

The experimental results of NLL tested and sampling times are saved in [`models/RNN/results.txt`](models/RNN).

### Transformer model

```bash
cd "models/Transformer"

python aqt.py
```

The experimental results of NLL tested and sampling times are saved in [`models/Transformer/results.txt`](models/Transformer).

### Argmax flow model

```bash
cd "models/argmax_flows"

python ArgMax.py
```

The experimental results of NLL tested and sampling times are saved in [`models/argmax_flows/results.txt`](models/argmax_flows).

### Discrete Denoising Flows (DDF) model

```bash
cd "models/DDF"

python DDF.py
```

The experimental results of NLL tested and sampling times are saved in [`models/DDF/results.txt`](models/DDF).

### Discrete Tree Flows (DTF) model

```bash
cd "models/DTF"

python DTF.py
```

The experimental results of NLL tested and sampling times are saved in [`models/DTF/results.txt`](models/DTF).

## **Acknowledgments**

This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on: [POVM_GENMODEL](https://github.com/carrasqu/POVM_GENMODEL), [QST-NNGMs-FNN](https://github.com/foxwy/QST-NNGMs-FNN), [argmax_flows](https://github.com/didriknielsen/argmax_flows), [Discrete-Denoising-Flows](https://github.com/alex-lindt/Discrete-Denoising-Flows), [Discrete-Tree-Flows](https://github.com/inouye-lab/Discrete-Tree-Flows).

## **License**

This code is distributed under an [Mozilla Public License Version 2.0](LICENSE).

Note that our code depends on other libraries, including POVM_GENMODEL, qMLE, and uses algorithms that each have their own respective licenses that must also be followed.
