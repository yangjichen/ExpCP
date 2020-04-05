# ExpCP
We develop a new tensor completion method called ExpCP for non-Gaussian data from the exponential family distributions. Here we mainly present code logic and data acquisition methods.

### Simulation and Real Dataset Study

1. The simulation experiment is divided into five parts, which are recorded in five folders, and the data generation method is also reserved.
2. The data used in the GDELT experiment is too large, so we only kept the code for data acquisition and collation. Divided into five stages to achieve our experiment.

### Core code

1. Thanks to Hancheng Ge's work. The pyten they wrote provided great convenience for the realization of our work, because they realized the basic operation of tensors. https://github.com/hanchengge/Helios-PyTen

2. Our contributions: pyten -> method -> ***BernoulliAirCP.py & PoissonAirCP.py & Neg_binAirCP.py***.It will be further optimized in the future and packaged in a file.



**Thanks to Professor Nan Zhang for his guidance and suggestions.** 