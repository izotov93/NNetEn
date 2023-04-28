[![License](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://img.shields.io/badge/DOI-arxiv-green)](https://arxiv.org/abs/2303.17995)

# Neural Network Entropy (NNetEn)

Entropy measures are effective features for time series classification problems. Traditional entropy measures, such as Shannon entropy, use probability distribution function. However, for the effective separation of time series, new entropy estimation methods are required to characterize the chaotic dynamic of the system. Our concept of Neural Network Entropy (NNetEn) is based on the classification of special datasets (MNIST-10 and SARS-CoV-2-RBV1) in relation to the entropy of the time series recorded in the reservoir of the LogNNet neural network. NNetEn estimates the chaotic dynamics of time series in an original way. Based on the NNetEn algorithm, we propose two new classification metrics: R2 Efficiency and Pearson Efficiency. 

## Article
The published article can be found at the link. [Manuscript](https://arxiv.org/abs/2303.17995 "arxiv.org")

##### Bibliographic link:

## Dataset
You can separately download the used datasets from the links [MNIST-10](https://yann.lecun.com/exdb/mnist/) and
[SARS-CoV-2-RBV1](https://data.mendeley.com/datasets/8hdnzv23x7)

## Installation

Installation is done from pypi using the following command

```shell
pip install NNetEn
```

## Usage

### Command to create the NNetEn_entropy model
```shell
 from NNetEn import NNetEn_entropy

 NNetEn = NNetEn_entropy(database='D1', mu=1)
```
Arguments:
- database: (default = D1) Select dataset, D1: MNIST, D2 :SARS-CoV-2-RBV1
- mu: (default = 1) Usage fraction of the selected database (0.01 .. 1).

**Output:** The LogNNet neural network is operated using normalized training and test
sets contained in the NNetEn_entropy class

### Command to calculation the NNetEn parameter
```shell
NNetEn.calculation(time_series, epoch=20, method=3, metric='Acc', log=False)
```
Arguments:
- time_series: Input data with a time series in numpy array format.
- epoch: (default = 20) The number of training epochs for the LogNNet neural
network, with a number greater than 0.
- method: (default = 3) One of 6 methods for forming a reservoir matrix from
the time series M1 ... M6.
- metric: (default = 'Acc') 'Acc' - accuracy metric,
                    'R2E' - R2 Efficiency metric,
                    'PE' - Pearson Efficiency metric.
- log: (default = False) Parameter for logging the main data used in the calculation.
Recording is done in log.txt file.

**Output:** Entropy value NNetEn.