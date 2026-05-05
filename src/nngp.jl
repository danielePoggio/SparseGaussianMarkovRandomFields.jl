function _fast_distance_computation(
    points::AbstractMatrix{F},
    i::Int,
    j::Int
)::F where {F <: Real}
    d = 0.0
    num_dim = size(points, 2)
    @inbounds @simd for k in 1:num_dim
        d += (points[i, k] - points[j, k])^2
    end
    return sqrt(d)
end

function _compute_AD!(
    strategy::S,
    variance::F,
    rho::F
) where {F <: Real, S <: AbstractStrategy}
    max_neighs = strategy.num_neighbors
    C_corr_cache = zeros(F, max_neighs, max_neighs)
    C_vec_cache = zeros(F, max_neighs)
    A_cache = zeros(F, max_neighs)
    num_points = size(strategy.points, 1)

    v_idx = 1
    strategy.D[1] = variance
    for i in 2:num_points
        C_corr_cache .= 0.0
        C_vec_cache .= 0.0
        A_cache .= 0.0
        Ni = strategy.neighbors[i]
        num_neighs_i = length(Ni)
        C_mat = view(C_corr_cache, 1:num_neighs_i, 1:num_neighs_i)
        c_vec = view(C_vec_cache, 1:num_neighs_i)
        a_vec = view(A_cache, 1:num_neighs_i)
        for j in 1:num_neighs_i
            idxj = Ni[j]
            dij = _fast_distance_computation(strategy.points, i, idxj)
            c_vec[j] = variance * exp(- rho * dij)
            C_mat[j, j] = variance
            for k in (j + 1):num_neighs_i
                idxk = Ni[k]
                djk = _fast_distance_computation(strategy.points, idxj, idxk)
                C_mat[j, k] = variance * exp(- rho * djk)
                C_mat[k, j] = C_mat[j, k]
            end
        end

        a_vec .= c_vec
        chol_C = cholesky!(Symmetric(C_mat))
        ldiv!(chol_C, a_vec)
        strategy.D[i] = variance - dot(a_vec, c_vec)
        for j in 1:num_neighs_i
            strategy.V[v_idx] = a_vec[j]
            v_idx += 1
        end
    end
end

function _quadratic_form(
    x::AbstractVector{F},
    strategy::S,
    u_buffer::AbstractVector{F}
) where {F <: Real, S <: AbstractStrategy}
    # Rimosse tutte le view inutili
    num_points = size(strategy.points, 1)
    
    @inbounds @simd for i in 1:num_points
        u_buffer[i] = x[i]
    end

    @inbounds @simd for k in eachindex(strategy.V)
        row = strategy.I[k]
        col = strategy.J[k]
        val = strategy.V[k]
        u_buffer[row] -= val * x[col]
    end

    q = zero(F)
    @inbounds @simd for i in 1:num_points
        q += (u_buffer[i]^2) / strategy.D[i]
    end
    
    return q
end

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