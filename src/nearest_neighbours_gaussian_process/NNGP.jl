module NNGP
export NNGP

using Random
using Distributions
using LinearAlgebra

include("strategy.jl")
include("utils.jl")


struct NearestNeighbourGaussianProcess{F <: Real, S <: AbstractStrategy} <: Distributions.ContinuousMultivariateDistribution
    strategy::S
    variance::F
    rho::F
    buffer::Vector{F}
end

function NearestNeighbourGaussianProcess(
    strategy::S,
    variance::F,
    rho::F
) where {F <: Real, S <: AbstractStrategy}
    buffer = zeros(F, size(strategy.points, 1))
    _compute_AD!(strategy, variance, rho)
    return NearestNeighbourGaussianProcess{F, S}(strategy, variance, rho, buffer)
end

Base.length(dist::NearestNeighbourGaussianProcess) = size(dist.strategy.points, 1)

function Distributions._rand!(
    rng::AbstractRNG,
    dist::NearestNeighbourGaussianProcess{F, S},
    x::AbstractVector{F}
) where {F <: Real, S <: AbstractStrategy}
    # Rimosse tutte le view inutili
    @inbounds @simd for i in eachindex(x)
        x[i] = sqrt(dist.strategy.D[i]) * randn(rng, F)
    end

    @inbounds @simd for vidx in eachindex(dist.strategy.V)
        x[dist.strategy.I[vidx]] += dist.strategy.V[vidx] * x[dist.strategy.J[vidx]]
    end
    x .= x[dist.strategy.inv_permutation]
    return x
end

function Distributions._logpdf(
    dist::NearestNeighbourGaussianProcess{F, S},
    x::AbstractVector{F}
)::F where {F <: Real, S <: AbstractStrategy}
    num_points = size(dist.strategy.points, 1)
    
    logdeterminant = sum(log, dist.strategy.D) 
    x .= x[dist.strategy.permutation]
    res = -0.5 * num_points * log(2 * pi) - 0.5 * logdeterminant - 0.5 * _quadratic_form(x, dist.strategy, dist.buffer)
    x .= x[dist.strategy.inv_permutation]
    return res
end

export NearestNeighbourGaussianProcess, MaximinOrderingStrategy

end