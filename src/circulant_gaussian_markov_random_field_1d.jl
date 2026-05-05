# ----------------------------------------------------------------
# Util functions
# ----------------------------------------------------------------

"""
Compute the eigenvalues for a circulant matrix with coefficient delta and offset.

"""
function _compute_circulant_eigenvalues(delta::F, n::Int; offset::F = 2.0)::Vector{F} where {F <: Real}
    eigenvalues = zeros(F, n)
    c0 = delta + offset
    for j in 1:n
        j_prime = j - 1
        eigenvalues[j] = c0 - 2 * cos(2 * pi * j_prime / n)
    end

    return eigenvalues
end

function _compute_circulant_precision_matrix!(
    Q::AbstractMatrix{F},
    delta::F;
    offset::F = 2.0
) where {F <: Real}

    n = size(Q, 1)
    c0 = delta + offset
    Q[1, 1] = c0
    Q[1, 2] = -1.0
    Q[1, n] = -1.0
    for i in 2:(n - 1)
        Q[i, i - 1] = -1.0
        Q[i, i] = c0
        Q[i, i + 1] = -1.0
    end
    Q[n, 1] = -1.0
    Q[n, n - 1] = -1.0
    Q[n, n] = c0
end

# ----------------------------------------------------------------
# Circulant Gaussian Markov Random Field
# ----------------------------------------------------------------
struct CirculantGaussianMarkovRandomField1D{F <: Real} <: Distributions.ContinuousMultivariateDistribution
    n::Int
    delta::F
    theta::F
    offset::F
    marginal_variance::F
end

# ------------------
# Constructors
# ------------------

function CirculantGaussianMarkovRandomField1D(
    n::Int,
    delta::F;
    offset::F = 2.0
) where {F <: Real}
    theta = acosh(0.5 * (delta + offset))
    marginal_var = coth(0.5 * n * theta) / (2.0 * sinh(theta))
    return CirculantGaussianMarkovRandomField1D{F}(n, delta, theta, offset, marginal_var)
end

Base.eltype(::CirculantGaussianMarkovRandomField1D{F}) where {F} = F
Base.length(dist::CirculantGaussianMarkovRandomField1D) = dist.n

function Distributions._rand!(
    rng::AbstractRNG,
    dist::CirculantGaussianMarkovRandomField1D,
    x::AbstractVector{F}
) where {F <: Real}
    eigenvalues = _compute_circulant_eigenvalues(dist.delta, dist.n, offset = dist.offset)
    eigenvalues .*= dist.marginal_variance
    eigenvalues .= 1.0 ./ sqrt.(eigenvalues)
    z_scaled = [complex(randn(rng), randn(rng)) * eigenvalues[i] for i in 1:dist.n]
    v = fft(z_scaled) / sqrt(dist.n)
    x .= real.(v)

    return x
end

function _quadratic_form(d::CirculantGaussianMarkovRandomField1D, x::AbstractVector{<:Real})
    res = zero(eltype(x))
    c0 = d.delta + d.offset
    n = d.n
    
    res += x[1]^2 * c0 - x[1] * x[2] - x[1] * x[n]
    for i in 2:(n - 1)
        res += x[i]^2 * c0 - x[i] * x[i - 1] - x[i] * x[i + 1]
    end
    res += x[n]^2 * c0 - x[n] * x[1] - x[n] * x[n - 1]
    
    return res * d.marginal_variance
end

function Distributions._logpdf(d::CirculantGaussianMarkovRandomField1D, x::AbstractVector{<:Real})
    # Log-Determinante di Q (somma dei logaritmi degli autovalori)
    eigenvals = _compute_circulant_eigenvalues(d.delta, d.n, offset=d.offset)
    eigenvals .*= d.marginal_variance
    log_det_Q = sum(log, eigenvals)
    
    # Log-Likelihood = 1/2 * log(|Q|) - n/2 * log(2π) - 1/2 * x^T Q x
    term1 = 0.5 * log_det_Q
    term2 = -0.5 * d.n * log(2π)
    term3 = -0.5 * _quadratic_form(d, x)
    
    return term1 + term2 + term3
end

Distributions.var(d::CirculantGaussianMarkovRandomField1D) = ones(d.n)
mean(d::CirculantGaussianMarkovRandomField1D) = zeros(d.n)

function Distributions.invcov(d::CirculantGaussianMarkovRandomField1D)
    Q = zeros(d.n, d.n)
    # Chiama la tua funzione in-place (che userà d.delta e d.offset)
    _compute_circulant_precision_matrix!(Q, d.delta, offset=d.offset)
    Q .*= d.marginal_variance
    return Q
end

function circulant_correlation(d::CirculantGaussianMarkovRandomField1D, k::Int)
    theta = acosh(0.5 * (d.delta + d.offset))
    k_prime = min(abs(k), abs(d.n - k))
    return cosh(theta * (0.5 * d.n - k_prime)) / cosh(0.5 * theta * d.n)
end

function Distributions.cov(d::CirculantGaussianMarkovRandomField1D)::Matrix
    C = zeros(d.n, d.n)
    for i in 1:d.n
        C[i , i] = 1.0
        for j in 1:(i - 1)
            C[i, j] = circulant_correlation(d, i - j)
            C[j, i] = C[i, j]
        end
    end
    return C
end