# GaussianMarkovRandomFields

[![Build Status](https://github.com/danielePoggio/GaussianMarkovRandomFields.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/danielePoggio/GaussianMarkovRandomFields.jl/actions/workflows/CI.yml?query=branch%3Amaster)

`GaussianMarkovRandomFields.jl` is a Julia package designed for the efficient evaluation and sampling of Gaussian Markov Random Fields (GMRFs) and Nearest Neighbour Gaussian Processes (NNGPs). 

It leverages sparse matrix operations and banded Cholesky factorizations to provide fast and scalable tools for spatial statistics and stochastic partial differential equation (SPDE) approaches.

## 📦 Installation

The package is currently in the process of being registered in the General registry. Once registered, you can install it using Julia's package manager. Open the Julia REPL, type `]` to enter the Pkg prompt, and run:

```julia
pkg> add GaussianMarkovRandomFields
```

## ✨ Features

* **Circulant GMRFs (1D):** Fast sampling and evaluation using Fast Fourier Transforms (FFT).
* **SPDE Matérn Fields:** Implementation of the SPDE approach for Matérn covariance functions using Delaunay triangulations and rigid mass/stiffness matrices.
* **Nearest Neighbour Gaussian Processes (NNGPs):** Scalable approximations for large spatial datasets.
* **Ordering Strategies:** Includes advanced node-ordering strategies like `MaximinOrderingStrategy` and `DelaunayOrderingStrategy` utilizing Reverse Cuthill-McKee (SymRCM) permutations for optimal bandwidth reduction.

## 🚀 Quick Start

Here is a simple example of how to create and sample from a 1D Circulant Gaussian Markov Random Field:

```julia
using GaussianMarkovRandomFields
using Random
using Distributions

# Define a 1D Circulant GMRF with 100 nodes and a delta parameter of 1.5
n_nodes = 100
delta = 1.5
dist = CirculantGaussianMarkovRandomField1D(n_nodes, delta)

# Pre-allocate a vector and sample from the distribution
rng = MersenneTwister(42)
x = zeros(n_nodes)
rand!(rng, dist, x)

# Evaluate the log-pdf of the generated sample
log_likelihood = logpdf(dist, x)
println("Log-likelihood of the sample: ", log_likelihood)
```

Here an example of how to create and sample from a NNGP in 2D with exponential covariance function

```julia
using Random
using Distributions
using GaussianMarkovRandomFields

# Generate a grid of points in [0, 1] × [0, 1]
nx, ny = 10, 10
x_sequence = range(0, 1, length=nx+1)
y_sequence = range(0, 1, length=ny+1)
n_nodes = (nx + 1) * (ny + 1)
points = zeros(n_nodes, 2)
points[:, 1] = repeat(x_sequence, inner = ny + 1)
points[:, 2] = repeat(y_sequence, outer = nx + 1)

# Define the strategy and create the distribution with variance 1.0 and spatial range 1.5
n_neighs = 10
variance = 1.0
rho = 1.5
strategy = MaximinOrderingStrategy(points, n_neighs)
dist = NearestNeighbourGaussianProcess(strategy, variance, rho)

# Pre-allocate a vector and sample from the distribution
rng = MersenneTwister(42)
x = zeros(n_nodes)
rand!(rng, dist, x)

# Evaluate the log-pdf of the generated sample
log_likelihood = logpdf(dist, x)
println("Log-likelihood of the sample: ", log_likelihood)

```

## 📚 References

The algorithms and mathematical frameworks implemented in this package are heavily based on the following seminal works:

1. Lindgren, F., Rue, H. and Lindström, J. (2011), *An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73: 423-498. https://doi.org/10.1111/j.1467-9868.2011.00777.x
2. Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications (1st ed.)*. Chapman and Hall/CRC. https://doi.org/10.1201/9780203492024
3. Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (2016). *Hierarchical Nearest-Neighbor Gaussian Process Models for Large Geostatistical Datasets* . Journal of the American Statistical Association, 111(514), 800–812. https://doi.org/10.1080/01621459.2015.1044091
4. Finley, A. O., Datta, A., Cook, B. D., Morton, D. C., Andersen, H. E., & Banerjee, S. (2019). *Efficient Algorithms for Bayesian Nearest Neighbor Gaussian Processes*. Journal of Computational and Graphical Statistics, 28(2), 401–414. https://doi.org/10.1080/10618600.2018.1537924

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
