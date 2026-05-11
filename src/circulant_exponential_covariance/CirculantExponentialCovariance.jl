module CirculantExponentialCovariance

using Random
using Distributions
using LinearAlgebra
using SparseArrays
using StatsFuns
using SpecialFunctions
using FFTW
using ForwardDiff

import ..SparseGaussianMarkovRandomFields: AbstractGaussianMarkovRandomField, AbstractCache

mutable struct CirculantExponentialCache{F <: Real} <: AbstractCache
    eigenvalues::Vector{F}
end


struct CirculantExponentialGaussianProcess{F <: Real, C <: CirculantExponentialCache} <: AbstractGaussianMarkovRandomField
    n::Int
    phi::F
    variance::F
    cache::C

end


function CirculantExponentialGaussianProcess(
    n::Int,
    phi::F,
    variance::F
) where {F <: Real}
    CirculantExponentialGaussianProcess(n, phi, variance, CirculantExponentialCache(zeros(F, n)))
end

Base.eltype(::CirculantExponentialGaussianProcess{F, C}) where {F, C} = F
Base.length(d::CirculantExponentialGaussianProcess) = d.n

"""
Computing eigenvalues for the circulant exponential covariance matrix using the DCT since the matrix is symmetric and positive definite.

"""

function _fft_symmetric(x::AbstractVector{<:Real})
    return real.(fft(x))
end

function _fft_symmetric(x::AbstractVector{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    valori = ForwardDiff.value.(x)
    fft_valori = real.(fft(valori))

    fft_derivate = zeros(V, length(x), N)
    for i in 1:N
        derivata_i = map(el -> ForwardDiff.partials(el, i), x)
        fft_derivate[:, i] = real.(fft(derivata_i))
    end
    
    risultato = similar(x)
    for j in 1:length(x)
        derivate_j = Tuple(fft_derivate[j, i] for i in 1:N)
        risultato[j] = ForwardDiff.Dual{T, V, N}(fft_valori[j], ForwardDiff.Partials(derivate_j))
    end
    
    return risultato
end

function _compute_circulant_eigenvalues_covariance!(
    eigenvalues::Vector{F},
    phi::F,
    variance::F
) where {F <: Real}
    n = length(eigenvalues)
    @inbounds for i in 1:n
        eigenvalues[i] = variance * exp(-phi * min(i - 1, n - i + 1))
    end

    eigenvalues .= _fft_symmetric(eigenvalues)

    return eigenvalues
end

function _compute_circulant_eigenvalues_covariance(
    n::Int,
    phi::F,
    variance::F
)::Vector{F} where {F <: Real}

    eigenvalues = zeros(F, n)
    _compute_circulant_eigenvalues_covariance!(eigenvalues, phi, variance)

    return eigenvalues
end

function Distributions._rand!(
    rng::AbstractRNG,
    dist::CirculantExponentialGaussianProcess{F, C},
    x::AbstractVector{F}
) where {F <: Real, C <: CirculantExponentialCache}
    n = length(dist)
    eigenvalues = zeros(F, n)
    _compute_circulant_eigenvalues_covariance!(eigenvalues, dist.phi, dist.variance)
    z = [complex(randn(rng), randn(rng)) for i in 1:dist.n]
    z .= sqrt.(eigenvalues) .* z
    @inbounds for i in 1:n
        x[i] = real(z[i])
    end
    return x
end

function Distributions._rand!(
    rng::AbstractRNG,
    dist::CirculantExponentialGaussianProcess{Float64, C},
    x::AbstractMatrix{Float64}
) where {C <: CirculantExponentialCache}
    n = length(dist)
    eigenvalues = dist.cache.eigenvalues
    _compute_circulant_eigenvalues_covariance!(eigenvalues, dist.phi, dist.variance)
    z = [complex(randn(rng), randn(rng)) for i in 1:dist.n]
    z .= sqrt.(eigenvalues) .* z
    @inbounds for i in 1:n
        x[i] = real(z[i])
    end
    return x
end

function Distributions._logpdf(
    dist::CirculantExponentialGaussianProcess{F, C},
    x::AbstractVector{<:Real}
) where {F <: Real, C <: CirculantExponentialCache}
    n = length(dist)
    eigenvalues = zeros(F, n)
    _compute_circulant_eigenvalues_covariance!(eigenvalues, dist.phi, dist.variance)
    fftx = fft(x)
    res = zero(F)
    @inbounds for i in 1:n
        quad_form = real(fftx[i] * conj(fftx[i])) / (n * eigenvalues[i])
        res += - 0.5 * log(eigenvalues[i]) - 0.5 * quad_form
    end
    res += - 0.5 * n * log(2 * pi)
    return res
end


function Distributions._logpdf(
    dist::CirculantExponentialGaussianProcess{Float64, C},
    x::AbstractMatrix{Float64}
) where {C <: CirculantExponentialCache}
    n = length(dist)
    eigenvalues = dist.cache.eigenvalues
    _compute_circulant_eigenvalues_covariance!(eigenvalues, dist.phi, dist.variance)
    fftx = fft(x)
    res = zero(Float64)
    @inbounds for i in 1:n
        res += - 0.5 * log(eigenvalues[i]) - 0.5 * fftx[i] * conj(fftx[i]) / eigenvalues[i]
    end
    return res
end



end