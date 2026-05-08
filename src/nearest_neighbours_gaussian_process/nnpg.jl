struct NearestNeighbourGaussianProcess{F <: Real, S <: AbstractStrategy} <: AbstractGaussianMarkovRandomField
    n::Int
    strategy::S
    variance::F
    rho::F
    buffer::Vector{F}
end

# 1. Costruttore standard (per Float64, utile per rand!)
function NearestNeighbourGaussianProcess(
    strategy::S,
    variance::Float64,
    rho::Float64
) where {S <: AbstractStrategy}
    
    buffer = zeros(Float64, size(strategy.points, 1))
    # Qui possiamo aggiornare strategy.V e strategy.D perché sono tutti Float64
    _compute_AD!(strategy.V, strategy.D, strategy.neighbors, strategy.points, variance, rho)
    
    n = size(strategy.points, 1)
    return NearestNeighbourGaussianProcess{Float64, S}(n, strategy, variance, rho, buffer)
end

# 2. Costruttore AD (per i Dual di Optim/Turing)
function NearestNeighbourGaussianProcess(
    strategy::S,
    variance::F,
    rho::F
) where {F <: Real, S <: AbstractStrategy} # F è un Dual
    
    buffer = zeros(F, size(strategy.points, 1))
    n = size(strategy.points, 1)
    
    # SALTAMO la chiamata a _compute_AD! qui dentro.
    # Non calcoliamo nulla, tanto se ne occuperà la tua _logpdf a zero allocazioni!
    
    return NearestNeighbourGaussianProcess{F, S}(n, strategy, variance, rho, buffer)
end

function Distributions._rand!(
    rng::AbstractRNG,
    dist::NearestNeighbourGaussianProcess{F, S},
    x::AbstractVector{F}
) where {F <: Real, S <: AbstractStrategy}
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
    x::AbstractVector{T} # I dati 'x' hanno il loro tipo T indipendente!
) where {F <: Real, S <: AbstractStrategy, T <: Real}
    
    num_points = size(dist.strategy.points, 1)
    num_nonzeros = length(dist.strategy.I)
    
    # Allochiamo D e V col tipo F di dist (così accolgono i Dual)
    D = zeros(F, num_points)
    V = zeros(F, num_nonzeros)
    
    _compute_AD!(V, D, dist.strategy.neighbors, dist.strategy.points, dist.variance, dist.rho)
    
    logdeterminant = sum(log, D) 
    x_permuted = x[dist.strategy.permutation]
    
    # La forma quadratica ora accetta x_permuted (Float) e V/D (Dual) senza protestare
    quad = _quadratic_form(x_permuted, dist.strategy.neighbors, V, D)
    
    res = -0.5 * num_points * log(2 * pi) - 0.5 * logdeterminant - 0.5 * quad
    
    return res
end

function logpdf_nograd(dist::NearestNeighbourGaussianProcess{F, S}, x::AbstractVector{F}) where {F <: Real, S <: AbstractStrategy}
    num_points = size(dist.strategy.points, 1)
    logdeterminant = sum(log, dist.strategy.D) 
    res = -0.5 * num_points * log(2 * pi) - 0.5 * logdeterminant - 0.5 * _quadratic_form(x, dist.strategy.neighbors, dist.strategy.V, dist.strategy.D)
    return res
end

# 1. Funzione interna privata: fa solo il "lavoro sporco" numerico a zero allocazioni.
function update_precision_values!(d::NearestNeighbourGaussianProcess{F}) where {F <: Real}
    # Aggiorna il V_Q interno usando i vettori D e V correnti
    _compute_precision_values!(
        d.strategy.V_Q, 
        d.strategy.V, 
        d.strategy.D, 
        d.strategy.neighbors
    )
end

# 2. La funzione principale (allocante) per l'utente, con la tua idea dello switch "sparse"
function Distributions.invcov(
    d::NearestNeighbourGaussianProcess{F};
    sparse_output::Bool = true
) where {F <: Real}
    
    # Aggiorniamo i valori interni
    update_precision_values!(d)
    
    # Assembliamo la matrice sparsa nel "mondo interno"
    Q_internal = SparseArrays.sparse(d.strategy.I_Q, d.strategy.J_Q, d.strategy.V_Q, d.n, d.n)
    
    # La permutiamo correttamente in 2D usando righe e colonne!
    inv_p = d.strategy.inv_permutation
    Q_external = Q_internal[inv_p, inv_p]
    
    # Restituiamo il formato richiesto
    if sparse_output
        return Q_external
    else
        return Matrix(Q_external)
    end
end

# 3. La versione in-place per quando l'utente ha GIÀ allocato una matrice Densa Q
function invcov!(
    Q::Matrix{F},
    d::NearestNeighbourGaussianProcess{F}
) where {F <: Real}
    
    # Chiama la nostra funzione di sopra chiedendo espressamente una matrice sparsa
    Q_sparse_external = invcov(d, sparse_output=true)
    
    # Copia i valori nella matrice densa dell'utente
    Q .= Q_sparse_external
    
    return Q
end