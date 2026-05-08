mutable struct Circulant1DCache{F <: Real} <: AbstractCache
    eigenvalues::Vector{F}
end


# ----------------------------------------------------------------
# Circulant Gaussian Markov Random Field
# ----------------------------------------------------------------
struct CirculantGaussianMarkovRandomField1D{F <: Real, C <: Circulant1DCache} <: AbstractGaussianMarkovRandomField
    n::Int
    delta::F
    theta::F
    offset::F
    marginal_variance::F
    cache::C
end

# ------------------
# Constructors
# ------------------

function CirculantGaussianMarkovRandomField1D(
    n::Int,
    delta::F;
    offset::Real = 2.0  # <--- Rilassiamo il tipo qui
) where {F <: Real}
    
    # 1. Convertiamo offset nello stesso tipo di delta (es. Dual)
    offset_F = convert(F, offset)
    
    # 2. Usiamo offset_F nei calcoli
    theta = acosh(0.5 * (delta + offset_F))
    marginal_var = coth(0.5 * n * theta) / (2.0 * sinh(theta))
    buffer = Circulant1DCache(zeros(F, n))
    
    return CirculantGaussianMarkovRandomField1D(n, delta, theta, offset_F, marginal_var, buffer)
end

Base.eltype(::CirculantGaussianMarkovRandomField1D{F}) where {F} = F

function Distributions._rand!(
    rng::AbstractRNG,
    dist::CirculantGaussianMarkovRandomField1D,
    x::AbstractVector{F}
) where {F <: Real}
    _compute_circulant_eigenvalues_1d!(dist.cache.eigenvalues, dist.delta, dist.n, offset = dist.offset)
    dist.cache.eigenvalues .*= dist.marginal_variance
    dist.cache.eigenvalues .= 1.0 ./ sqrt.(dist.cache.eigenvalues)
    z_scaled = [complex(randn(rng), randn(rng)) * dist.cache.eigenvalues[i] for i in 1:dist.n]
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
    eigenvals = _compute_circulant_eigenvalues_1d(d.delta, d.n, offset=d.offset)
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

function invcov!(Q::AbstractMatrix{F}, d::CirculantGaussianMarkovRandomField1D{F}) where {F <: Real}
    # 1. Sovrascrive Q con la matrice di precisione base (chiamata in-place)
    _compute_circulant_precision_matrix_1d!(Q, d.delta, offset=d.offset)
    
    # 2. Scala tutti i valori in-place
    Q .*= d.marginal_variance
    
    return Q
end

function Distributions.invcov(d::CirculantGaussianMarkovRandomField1D)
    Q = zeros(d.n, d.n)
    invcov!(Q, d)
    return Q
end
