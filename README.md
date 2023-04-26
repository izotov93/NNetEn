[![License](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://img.shields.io/badge/DOI-arxiv-green)](https://arxiv.org/abs/2303.17995)

# Neural Network Entropy (NNetEn)

Entropy measures are effective features for time series classification problems. Traditional entropy measures, such as Shannon entropy, use probability distribution function. However, for the effective separation of time series, new entropy estimation methods are required to characterize the chaotic dynamic of the system. Our concept of Neural Network Entropy (NNetEn) is based on the classification of special datasets (MNIST-10 and SARS-CoV-2-RBV1) in relation to the entropy of the time series recorded in the reservoir of the LogNNet neural network. NNetEn estimates the chaotic dynamics of time series in an original way. Based on the NNetEn algorithm, we propose two new classification metrics: R2 Efficiency and Pearson Efficiency. 

##### [Manuscript](https://arxiv.org/abs/2303.17995 "arxiv.org")

## Installation

Installation is done from pypi using the following command

```shell
pip install NNetEn
```

## Usage

##### Command to create a NNetEn_entropy model
```shell
 from NNetEn import NNetEn_entropy

 NNetEn = NNetEn_entropy(database='D1', mu=1)
```
Arguments:
- database (default = D1) Select dataset, D1 – MNIST, D2 – SARS-CoV-2-RBV1
- mu (default = 1) usage fraction of the selected dataset μ (0.01…1).

**Output:** The LogNNet neural network is operated using normalized training and test
sets contained in the NNetEn_entropy class

##### Command to calculation a NNetEn parameter
```shell
NNetEn.calculation(time_series, epoch=20, method=3, metric=’Acc’, log=False)
```
Arguments:
- time_series - input data with a time series in numpy array format.
- epoch (default epoch = 20). The number of training epochs for the LogNNet neural
network, with a number greater than 0.
- method (default method = 3). One of 6 methods for forming a reservoir matrix from
the time series M1 ... M6.
- metric = (default metric = 'Acc'). Options: metric = 'Acc', metric = 'R2E', metric = 'PE'.
- log (default = False) Parameter for logging the main data used in the calculation.
Recording is done in log.txt file

**Output:** NNetEn – the entropy value for the given parameters.