
struct SPDEMesh{F <: Real} <: AbstractMesh
    C::AbstractMatrix{F}  # mass matrix
    G::AbstractMatrix{F}  # stiffness matrix
    D::AbstractMatrix{F}  # distance matrix
    invC::AbstractMatrix{F} # inverse mass matrix dumped, diagonal matrix to keep the sparsity
    permutation::AbstractVector{Int}
    inverse_permutation::AbstractVector{Int}
    p::Int # bandwidth
end

function SPDEMesh(points::AbstractMatrix{F}) where {F <: Real}
    num_points = size(points, 2)
    C = zeros(F, num_points, num_points)
    G = zeros(F, num_points, num_points)
    D = zeros(F, num_points, num_points)
    invC = zeros(F, num_points, num_points)
    tri = triangulate(points)
    C, G = compute_rigid_mass_matrices(tri);
    permutation, inverse_permutation = find_optimal_permutation!(C, G)
    points_permuted = points[:, permutation]
    p = find_bandwidth(C)
    for i in 1:num_points
        invC[i, i] = 1.0 / sum(view(C, i, :))
        for j in 1:(i - 1)
            D[i, j] = norm(points_permuted[:, i] - points_permuted[:, j])
            D[j, i] = D[i, j]
        end
    end
    return SPDEMesh(C, G, D, invC, permutation, inverse_permutation, p)
end


struct SPDEMatern{F <: Real} <: Distributions.ContinuousMultivariateDistribution
    kappa::F
    tau::F
    sigma2::F      # Varianza target desiderata
    alpha::Int     # Operatore SPDE
    d::Int
    mesh::SPDEMesh{F}
end

Base.length(d::SPDEMatern) = size(d.mesh.C, 1)

# Costruttore intelligente: NON passiamo tau, passiamo sigma2!
function SPDEMatern(kappa::F, sigma2::F, mesh::SPDEMesh{F}; alpha::Int = 2, d::Int = 2) where {F <: Real}
    nu = alpha - d / 2.0
    
    if nu <= 0
        error("Varianza infinita! Per d=2, alpha deve essere > 1")
    end

    # Calcolo inverso: troviamo il tau^2 che forza la varianza a essere esattamente sigma2
    tau_sq = exp(loggamma(nu) - loggamma(alpha) - (d/2.0)*log(4*pi) - 2.0*nu*log(kappa) - log(sigma2))
    tau = sqrt(tau_sq)
    
    return SPDEMatern(kappa, tau, sigma2, alpha, d, mesh)
end

# Restituisce Q rigorosamente a banda (nel mondo permutato)
function _precision_matrix_permuted(d::SPDEMatern{F})::Matrix{F} where {F <: Real}
    n = size(d.mesh.C, 1)
    
    alpha = d.alpha
    if alpha == 2.0
        # 1. Costruiamo K
        K = (d.kappa^2) .* d.mesh.C .+ d.mesh.G

        # 2. A = D^{1/2} K
        A = zeros(F, n, n)
        for i in 1:n
            A[i, :] .= K[i, :] .* sqrt(d.mesh.invC[i, i])
        end

        # 3. K C^{-1} K
        precision_matrix = A' * A
        
        # 4. LA SCALATURA CORRETTA
        precision_matrix .*= (d.tau^2)
    else
        @warn "SPDEMatern not implemented for alpha != 2.0"
        precision_matrix = zeros(F, n, n)
    end
    
    return precision_matrix
end

# Restituisce L rigorosamente a banda (nel mondo permutato)
function _compute_cholesky_permuted(d::SPDEMatern{F})::Matrix{F} where {F <: Real}
    Q_perm = _precision_matrix_permuted(d)
    p = find_bandwidth(Q_perm)
    L_perm = zeros(F, size(Q_perm, 1), size(Q_perm, 2))
    
    compute_banded_cholesky!(L_perm, Q_perm, p)
    
    return L_perm
end

function Distributions.invcov(d::SPDEMatern{F})::Matrix{F} where {F <: Real}
    Q_perm = _precision_matrix_permuted(d)
    
    # Riordiniamo la matrice per l'utente, riportandola al dominio originale
    inv_p = d.mesh.inverse_permutation
    return Q_perm[inv_p, inv_p]
end


"""
Solve linear system U x = b where U is an upper triangular matrix with upper bandwidth p.
This is the Backward Substitution algorithm.
"""
function backward_substitution(
    U::AbstractMatrix{F}, 
    b::AbstractVector{F};
    p::Int = 0
) where {F <: Real}

    n = length(b)
    x = zeros(F, n)
    
    # Se p non è specificato, la banda è larga quanto la matrice
    if p == 0
        p = n
    end

    # Iteriamo dal basso verso l'alto (dalla riga n alla riga 1)
    @inbounds for i in n:-1:1
        # Limite destro della banda per la riga corrente
        col_end = min(n, i + p)
        
        # Calcoliamo il prodotto scalare solo sugli elementi fuori diagonale
        s = zero(F)
        for k in (i + 1):col_end
            s += U[i, k] * x[k]
        end
        
        # Sottraiamo dal termine noto b[i] e dividiamo per il pivot sulla diagonale
        x[i] = (b[i] - s) / U[i, i]
    end
    
    return x
end

function backward_substitution!(
    x::AbstractVector{F},
    U::AbstractMatrix{F}, 
    b::AbstractVector{F},
    p::Int
) where {F <: Real}

    n = length(b)
    
    @inbounds for i in n:-1:1
        col_end = min(n, i + p)
        
        s = zero(F)
        for k in (i + 1):col_end
            s += U[i, k] * x[k]
        end
        
        x[i] = (b[i] - s) / U[i, i]
    end
    
    return x
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::SPDEMatern{F},
    x::AbstractVector{F}
) where {F <: Real}

    n = size(d.mesh.C, 1)
    
    # Lavoriamo con la L a banda permutata!
    L_perm = _compute_cholesky_permuted(d)
    
    b = randn(rng, F, n)
    x_perm = zeros(F, n) # Vettore temporaneo nel mondo permutato
    
    # Risolviamo nel mondo permutato
    backward_substitution!(x_perm, L_perm', b, d.mesh.p)
    
    # Mappiamo il risultato nel vettore x (mondo utente)
    x .= x_perm[d.mesh.inverse_permutation]
    
    return x
end

function Distributions._logpdf(
    d::SPDEMatern{F},
    x::AbstractVector{F}
) where {F <: Real}

    n = size(d.mesh.C, 1)
    
    # 1. Portiamo i dati dell'utente nel nostro mondo permutato
    x_permuted = x[d.mesh.permutation]
    
    # 2. Otteniamo la L a banda
    L_perm = _compute_cholesky_permuted(d)
    
    # 3. Termine 1: Costante (senza tau, è già in L)
    term1 = -0.5 * n * log(2 * π)
    
    # 4. Termine 2: Log-Determinante di L
    term2 = sum(log(L_perm[i, i]) for i in 1:n)
    
    # 5. Termine 3: Forma quadratica (usando i dati permutati)
    res = L_perm' * x_permuted
    term3 = -0.5 * dot(res, res)
    
    return term1 + term2 + term3
end
