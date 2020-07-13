# ExpCP
We develop a new tensor completion method called ExpCP for non-Gaussian data from the exponential family distributions. Here we mainly present code logic and data acquisition methods.

### Simulation and Real Dataset Study

1. The simulation experiment is divided into five parts, which are recorded in five folders, and the data generation method is also reserved.
2. The data used in the GDELT experiment is too large, so we only kept the code for data acquisition and collation. Divided into five stages to achieve our experiment.

### Core code

1. Thanks to Hancheng Ge's work. The pyten they wrote provided great convenience for the realization of our work, because they realized the basic operation of tensors. https://github.com/hanchengge/Helios-PyTen

2. Our contributions: pyten -> method -> ***BernoulliAirCP.py & PoissonAirCP.py & Neg_binAirCP.py***.It will be further optimized in the future and packaged in one file.

### Our paper
Tensor completion is among the most important tasks in tensor data analysis, which aims to fill the missing entries of a partially observed tensor. In many real applications, non‐Gaussian data such as binary or count data are frequently collected. Thus it is inappropriate to assume that observations are normally distributed, and formulate tensor completion with least squares based approach. In this article, we develop a new tensor completion method called ExpCP for non‐Gaussian data from the exponential family distributions. Under the framework of penalized likelihood estimation, we assume the tensor of latent natural parameters admits a low‐rank structure, and further induce regularization by integrating auxiliary information. Moreover, we propose an efficient algorithm based on the alternating direction method of multipliers to solve the optimization problem. We establish the algorithmic convergence results, and demonstrate the efficacy of our proposal with comprehensive simulations and real data analysis on GDELT social events study. The empirical results show that our method offers improved modeling strategy as well as efficient computation.


**Thanks to Professor Nan Zhang for his guidance and suggestions.** 
