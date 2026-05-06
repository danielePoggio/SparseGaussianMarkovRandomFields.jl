module SPDE
export SPDE

using DelaunayTriangulation
using Distributions
using Random
using LinearAlgebra
using SparseArrays

include("utils.jl")

struct SPDEMesh{F <: Real} <: AbstractMesh
    C::SparseMatrixCSC{F, Int}  # mass matrix
    G::SparseMatrixCSC{F, Int}  # stiffness matrix
    invC::Vector{F}             # vettore diagonale della inverse mass matrix lumped
    permutation::Vector{Int}
    inverse_permutation::Vector{Int}
    p::Int                      # bandwidth
end

function SPDEMesh(points::AbstractMatrix{F}) where {F <: Real}
    num_points = size(points, 2)
    
    tri = triangulate(points)
    
    # IMPORTANTE: Questa funzione DEVE restituire matrici sparse (SparseMatrixCSC)
    C, G = compute_rigid_mass_matrices(tri)
    
    permutation, inverse_permutation = find_optimal_permutation!(C, G)
    p = find_bandwidth(C)
    
    # Calcoliamo la lumped mass matrix (vettore invece di matrice densa NxN)
    invC = zeros(F, num_points)
    for i in 1:num_points
        # Somma lungo la riga di C per il "row-lumping"
        invC[i] = 1.0 / sum(view(C, i, :)) 
    end
    
    # Abbiamo rimosso D! Non serve per SPDE e causa problemi di memoria.
    return SPDEMesh(C, G, invC, permutation, inverse_permutation, p)
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

# Restituisce la matrice di precisione Q come matrice sparsa
function _precision_matrix_permuted(d::SPDEMatern{F}) where {F <: Real}
    n = size(d.mesh.C, 1)
    
    if d.alpha == 2
        # K è una matrice sparsa
        K = (d.kappa^2) .* d.mesh.C .+ d.mesh.G

        # Calcolo di K * C^{-1} * K sfruttando matrici sparse e diagonali
        Q = K * Diagonal(d.mesh.invC) * K
        Q .*= (d.tau^2)
        
        # Dichiariamo esplicitamente che è simmetrica per aiutare il solver
        return Symmetric(Q)
    else
        @warn "SPDEMatern not implemented for alpha != 2"
        return Symmetric(spzeros(F, n, n))
    end
end

function Distributions.invcov(d::SPDEMatern{F})::Matrix{F} where {F <: Real}
    # Per l'inversa, restituiamo una densa come si aspetta l'utente
    Q_perm = Matrix(_precision_matrix_permuted(d))
    
    inv_p = d.mesh.inverse_permutation
    return Q_perm[inv_p, inv_p]
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::SPDEMatern{F},
    x::AbstractVector{F}
) where {F <: Real}

    n = size(d.mesh.C, 1)
    Q_perm = _precision_matrix_permuted(d)
    
    # Fattorizzazione di Cholesky sparsa nativa
    # Nota: cholesky() applicherà un'ulteriore permutazione interna per ottimizzare la sparsità
    chol = cholesky(Q_perm) 
    
    # Campioniamo rumore bianco
    z = randn(rng, F, n)
    
    # Risolviamo il sistema U * y = z (sostituisce backward_substitution!)
    y = chol.U \ z 
    
    # Riordiniamo rispetto alla permutazione interna di SuiteSparse
    x_perm = zeros(F, n)
    x_perm[chol.p] .= y
    
    # Mappiamo il risultato nel vettore x dell'utente (invertendo la TUA permutazione RCM)
    @inbounds @simd for i in 1:n
        x[i] = x_perm[d.mesh.inverse_permutation[i]]
    end
    
    return x
end

function Distributions._logpdf(
    d::SPDEMatern{F},
    x::AbstractVector{F}
) where {F <: Real}

    n = size(d.mesh.C, 1)
    
    # Portiamo i dati dell'utente nel nostro mondo RCM permutato
    x_permuted = x[d.mesh.permutation]
    
    Q_perm = _precision_matrix_permuted(d)
    
    # Calcoliamo la fattorizzazione solo per ottenere il log-determinante
    chol = cholesky(Q_perm)
    
    term1 = -0.5 * n * log(2 * π)
    
    # logdet è una funzione nativa ottimizzata per matrici fattorizzate!
    term2 = 0.5 * logdet(chol) 
    
    # Forma quadratica x^T * Q * x calcolata in modo ultra-efficiente e sparso
    term3 = -0.5 * dot(x_permuted, Q_perm * x_permuted)
    
    return term1 + term2 + term3
end

end # module