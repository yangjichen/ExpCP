# ExpCP

We develop a new tensor completion method called ExpCP for non-Gaussian data from the exponential family distributions, and further induce regularization by integrating auxiliary information. Here we mainly present code logic and data acquisition methods. 

### Overview

Tensor completion is among the most important tasks in tensor data analysis, which aims to fill the missing entries of a partially observed tensor. In many real applications, non-Gaussian data such as binary or count data are frequently collected. Thus, it is inappropriate to assume that observations are normally distributed and formulate tensor completion with least squares based approaches. In this article, we develop a new tensor completion method called ExpCP for non-Gaussian data from the exponential family distributions. Under the framework of penalized likelihood estimation, we assume the tensor of latent natural parameters admits a low-rank structure and further induce regularization by integrating auxiliary information. Moreover, we propose an efficient algorithm based on the alternating direction method of multipliers to solve the optimization problem. We establish the algorithmic convergence results and demonstrate the efficacy of our proposal with comprehensive simulations and real data analysis on GDELT social events study. The empirical results show that our method offers improved modelling strategies as well as efficient computation.

### Prepare the environment

To run this code, you need to install the dependencies as follows,

```python
scipy==1.2.1
seaborn==0.9.0
numpy==1.16.4
matplotlib==3.0.2
pyspark==2.4.4
statsmodels==0.9.0
gdelt==0.1.10.6
pandas==0.23.4
scikit_learn==0.24.2
```

### Simulation 

The simulation experiment is divided into five parts, which are recorded in five folders, and the data generation method is also reserved.

`EXP1`: Focus on comparing different tensor-based state-of-the-art methods.

`EXP2`: Examine the elapsed time for each method on synthetic datasets.

`EXP3`: Investigate the sensitivity to the choices of the prespecified rank.

`EXP4`: Explore the impact of integrating auxiliary information.

`EXP5`: Examine the performance of the inexact algorithm and compare it with the exact version.

### Real Dataset Study

The data used in the GDELT experiment is too large, so we only kept the code for data acquisition and collation. Divided into five stages to achieve our experiment.

`GDELT1.py`: Download a subset of GDELT dataset which covers the period from January, 2018 to December, 2018.

`GDELT2.py`: Reorder the dataset as a tensor.

`GDELT3.py`: Compare the different tensor-based state-of-the-art methods.

`GDELT4.py`: Effects of auxiliary infomation.

`GDELT5.py`: Effects of Regularization Parameters.

### Reference:

For details of the algorithm and implementation, see our paper:

[1] [Jichen yang, Nan Zhang, 2020. "Exponential family tensor completion with auxiliary information". Stat 9, e296.](https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.296)


