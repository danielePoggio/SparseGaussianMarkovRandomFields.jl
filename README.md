# SparseGaussianMarkovRandomFields

[![Build Status](https://github.com/danielePoggio/GaussianMarkovRandomFields.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/danielePoggio/GaussianMarkovRandomFields.jl/actions/workflows/CI.yml?query=branch%3Amaster)

`SparseGaussianMarkovRandomFields.jl` is a Julia package designed for the efficient evaluation and sampling of Gaussian Markov Random Fields (GMRFs) and Nearest Neighbour Gaussian Processes (NNGPs). 

It is compatible with the default library `Distributions.jl` and `Turing.jl` via ForwardDiff. 

The introduction of *AbstractCache* object allows to stored all the information about the structures of the precision matrix (and preallocating it) in order to improve the performance on huge massively MCMC iteration, keeping the default pipeline of `Distributions.jl` setup. In order to avoid issue with AD algorithms, the computations of logpdf for these distributions, avoid to use the preallocated containers for the value of matrix precision, rebuilding it if it is necessary.

## Installation

The package is currently in the process of being registered in the General registry. Once registered, you can install it using Julia's package manager. Open the Julia REPL, type `]` to enter the Pkg prompt, and run:

```julia
pkg> add GaussianMarkovRandomFields
```

## Features

1. **Circulant GMRFs (1D):** Fast sampling and evaluation using Fast Fourier Transforms (FFT).
2. **Nearest Neighbour Gaussian Processes (NNGPs):** Scalable approximations for large spatial datasets.
3. **Ordering Strategies:** Includes advanced node-ordering strategies like `MaximinOrderingStrategy` and `DelaunayOrderingStrategy` utilizing Reverse Cuthill-McKee (SymRCM) permutations for optimal bandwidth reduction.

## Quick Start

Here is a simple example of how to create and sample from a 1D Circulant Gaussian Markov Random Field:

```julia
using SparseGaussianMarkovRandomFields
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
using SparseGaussianMarkovRandomFields

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

## NNGP Maximum Likehood Inference with `Optim.jl`
```julia
using ADTypes
using Optim
using Random
using Distributions
using LinearAlgebra
using SparseGaussianMarkovRandomFields

rng = MersenneTwister(42)

# Building grid and generating synthetic data 

nx, ny = 20, 20
x_sequence = range(0, 1, length=nx+1)
y_sequence = range(0, 1, length=ny+1)
n_nodes = (nx + 1) * (ny + 1)
points = zeros(n_nodes, 2)
points[:, 1] = repeat(x_sequence, inner = ny + 1)
points[:, 2] = repeat(y_sequence, outer = nx + 1)

max_distance = sqrt(2)
min_distance = x_sequence[2] - x_sequence[1]

rho_max = 3.0 / min_distance
rho_min = 3.0 / max_distance

# true hyperparameters
true_variance = 1.0
true_rho = 6.0

# defining the ordering strategy
n_neighs = 20
strategy = MaximinOrderingStrategy(points, n_neighs)

# create distribution object 
true_dist = NearestNeighbourGaussianProcess(strategy, true_variance, true_rho)
y = zeros(n_nodes)
rand!(rng, true_dist, y)

# function to minimize with Optim.jl
function neg_log_likelihood_nngp(params)
    var_val = exp(params[1])
    rho_val = exp(params[2])
    d = NearestNeighbourGaussianProcess(strategy, var_val, rho_val)
    return -logpdf(d, y)
end

lower_bounds = [-5.0, log(rho_min)]
upper_bounds = [5.0, log(rho_max)]
initial_guess = [log(0.5), log(0.5 * (rho_max + rho_min))]

mle_results = optimize(
    neg_log_likelihood_nngp, 
    lower_bounds, 
    upper_bounds, 
    initial_guess, 
    Fminbox(LBFGS()); 
    autodiff = AutoForwardDiff() # <--- ForwardDiff viene attivato da qui!
)

var_mle, rho_mle = exp.(Optim.minimizer(mle_results))

println("MLE Results:")
println("Estimated Variance: $(round(var_mle, digits=3))")
println("Estimated Rho:      $(round(rho_mle, digits=3))")
println("Iterations:         $(Optim.iterations(risultato_mle))\n")


```


## NNGP MCMC Inference with `Turing.jl` 
```julia

using Turing
using ForwardDiff
using Random
using Distributions
using LinearAlgebra
using SparseGaussianMarkovRandomFields

rng = MersenneTwister(42)

# Building grid and generating synthetic data 

nx, ny = 20, 20
x_sequence = range(0, 1, length=nx+1)
y_sequence = range(0, 1, length=ny+1)
n_nodes = (nx + 1) * (ny + 1)
points = zeros(n_nodes, 2)
points[:, 1] = repeat(x_sequence, inner = ny + 1)
points[:, 2] = repeat(y_sequence, outer = nx + 1)

max_distance = sqrt(2)
min_distance = x_sequence[2] - x_sequence[1]

rho_max = 3.0 / min_distance
rho_min = 3.0 / max_distance

# true hyperparameters
true_variance = 1.0
true_rho = 6.0

# defining the ordering strategy
n_neighs = 20
strategy = MaximinOrderingStrategy(points, n_neighs)

# create distribution object 
true_dist = NearestNeighbourGaussianProcess(strategy, true_variance, true_rho)
y = zeros(n_nodes)
rand!(rng, true_dist, y)

# define turing model
@model function nngp_model(y, strategy)
    variance ~ Gamma(2.0, 0.5)
    rho ~ Gamma(3.0, 0.5)
    dist = NearestNeighbourGaussianProcess(strategy, variance, rho)
    
    y ~ dist
end

# create model with observations
nngp_model = nngp_model(y, strategy)

# run the model using Turing interface
chain_nngp = sample(nngp_model, NUTS(0.65), 1000)

var_mcmc = mean(chain_nngp[:variance])
rho_mcmc = mean(chain_nngp[:rho])
var_quantile = quantile(chain_nngp[:variance], [0.025, 0.500, 0.975])
rho_quantile = quantile(chain_nngp[:rho], [0.025, 0.500, 0.975])

println("Parameter  | True  | MCMC (Posterior Mean) [CI 95%]")
println("--------------------------------------------------")
println("Variance   | $true_variance   | $(round(var_mcmc, digits=3)) [$(round(var_quantile[1], digits=3)) - $(round(var_quantile[3], digits=3))]")
println("Rho        | $true_rho   | $(round(rho_mcmc, digits=3)) [$(round(rho_quantile[1], digits=3)) - $(round(rho_quantile[3], digits=3))]\n")

```

## Exponential Covariance Maximum Likehood Inference with `Optim.jl`
```julia

using ADTypes
using Optim
using Turing
using Random
using Distributions
using SparseGaussianMarkovRandomFields

n_nodes = 24
phi_min = 3.0 / 24
phi_max = 3.0
true_phi = 2.5
true_variance = 1.0
rng = MersenneTwister(42)

dist_vera = CirculantExponentialGaussianProcess(n_nodes, true_phi, true_variance)
y = rand(rng, dist_vera)

function neg_log_likelihood(params)
    phi = exp(params[1]) 
    variance = exp(params[2])
    d = CirculantExponentialGaussianProcess(n_nodes, phi, variance)
    return -logpdf(d, y)
end

lower_bounds = [-5.0, log(phi_min)]
upper_bounds = [5.0, log(phi_max)]
initial_guess = [log(0.5), log(0.5 * (phi_max + phi_min))]

risultato_mle = optimize(
    neg_log_likelihood, 
    lower_bounds, 
    upper_bounds, 
    initial_guess, 
    Fminbox(LBFGS()); 
    autodiff = AutoForwardDiff()
)

phi_mle, variance_mle = exp.(Optim.minimizer(risultato_mle))

println("True Phi: $true_phi")
println("Estimated Phi (MLE): $(round(phi_mle, digits=3))\n")
println("True Variance: $true_variance")
println("Estimated Variance (MLE): $(round(variance_mle, digits=3))")
```

## Exponential Covariance MCMC Inference with `Turing.jl`

```julia

using Turing
using ForwardDiff
using Random
using Distributions
using SparseGaussianMarkovRandomFields

n_nodes = 24
phi_min = 3.0 / 24
phi_max = 3.0
true_phi = 2.5
true_variance = 1.0
rng = MersenneTwister(42)

dist_vera = CirculantExponentialGaussianProcess(n_nodes, true_phi, true_variance)
y = rand(rng, dist_vera)

@model function circulant_model(y, n_nodes)
    phi ~ Uniform(phi_min, phi_max) 
    variance ~ Gamma(1.0, 1.0)
    
    y ~ CirculantExponentialGaussianProcess(n_nodes, phi, variance)
end

modello = circulant_model(y, n_nodes)

chain = sample(modello, NUTS(0.65), 1000)

display(chain)

phi_vec = vec(chain[:phi])
variance_vec = vec(chain[:variance])
quantiles_phi = quantile(phi_vec, [0.025, 0.500, 0.975])
quantiles_variance = quantile(variance_vec, [0.025, 0.500, 0.975])


println("True Phi: $true_phi")
println("Estimated Phi (MCMC): $(round(quantiles_phi[2], digits=3))")
println("Credibility Interval 95% for Phi: $(round(quantiles_phi[1], digits=3)) - $(round(quantiles_phi[3], digits=3))")

println("True Variance: $true_variance")
println("Estimated Variance (MCMC): $(round(quantiles_variance[2], digits=3))")
println("Credibility Interval 95% for Variance: $(round(quantiles_variance[1], digits=3)) - $(round(quantiles_variance[3], digits=3))")

```

## References

The algorithms and mathematical frameworks implemented in this package are heavily based on the following seminal works:

1. Lindgren, F., Rue, H. and Lindström, J. (2011), *An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73: 423-498. https://doi.org/10.1111/j.1467-9868.2011.00777.x
2. Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications (1st ed.)*. Chapman and Hall/CRC. https://doi.org/10.1201/9780203492024
3. Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (2016). *Hierarchical Nearest-Neighbor Gaussian Process Models for Large Geostatistical Datasets* . Journal of the American Statistical Association, 111(514), 800–812. https://doi.org/10.1080/01621459.2015.1044091
4. Finley, A. O., Datta, A., Cook, B. D., Morton, D. C., Andersen, H. E., & Banerjee, S. (2019). *Efficient Algorithms for Bayesian Nearest Neighbor Gaussian Processes*. Journal of Computational and Graphical Statistics, 28(2), 401–414. https://doi.org/10.1080/10618600.2018.1537924

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
