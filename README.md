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

## 📚 References

The algorithms and mathematical frameworks implemented in this package are heavily based on the following seminal works:

1. Lindgren, F., Rue, H., & Lindström, J. (2011). *An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(4), 423-498.
2. Rue, H., & Held, L. (2005). *Gaussian Markov random fields: theory and applications*. CRC press.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.