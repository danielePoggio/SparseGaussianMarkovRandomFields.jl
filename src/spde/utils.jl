# ----------------------------------------------------------------
# Utils
# ----------------------------------------------------------------

compute_area(p, q, r) = 0.5 * ((getx(q) - getx(p)) * (gety(r) - gety(p)) - (gety(q) - gety(p)) * (getx(r) - getx(p)))


"""
Compute the mass matrix C and the rigidity matrix G for a triangulation 'tri' built using DelaunayTriangulation.jl.
The computation can be found in the paper "An explicit link between Gaussian Fields and Gaussian Markov Random Fields: the stochastic partial differential equation approach" by F. Lindgren and H. Rue (2011).

"""
function compute_rigid_mass_matrices(
    tri
)

    num_vertices = num_solid_vertices(tri)

    C = zeros(num_vertices, num_vertices)
    G = zeros(num_vertices, num_vertices)

    nodes = tri.points'

    local_C = zeros(3, 3)
    for i in 1:3
        for j in 1:3
            if i == j
                local_C[i, j] = 2.0
            else
                local_C[i, j] = 1.0
            end
        end
    end

    local_G = zeros(3, 3)

    local_E = zeros(2, 3)

    for t in each_solid_triangle(tri)
        global_v0 = t[1]
        global_v1 = t[2]
        global_v2 = t[3]

        global_indices = [global_v0, global_v1, global_v2]

        v0 = nodes[global_v0, :]
        v1 = nodes[global_v1, :]
        v2 = nodes[global_v2, :]

        e0 = v2 - v1
        e1 = v0 - v2
        e2 = v1 - v0

        local_E[:, 1] .= e0
        local_E[:, 2] .= e1
        local_E[:, 3] .= e2

        triangle_area = compute_area(v0, v1, v2)

        local_G .= 0.25 * local_E' * local_E / triangle_area

        C[global_indices, global_indices] .+= triangle_area * local_C / 12.0
        G[global_indices, global_indices] .+= local_G

    end

    return C, G

end

"""
Find the permutation that minimizes the bandwidth of a symmetric banded matrix C using Reverse Cuthill-McKee permutation.
Once the permutation is found, the matric of points, C (mass matrix) and G (stiffness matrix) are updated.

"""

function find_optimal_permutation!(
    C::AbstractMatrix{F},
    G::AbstractMatrix{F}
) where {F <: Real}

    permutation = symrcm(sparse(C))
    inverse_permutation = similar(permutation)
    inverse_permutation[permutation] = 1:length(permutation)
    C .= C[permutation, permutation]
    G .= G[permutation, permutation]
    return permutation, inverse_permutation
end


"""
Find the bandwidth of a symmetric banded matrix C.

"""

function find_bandwidth(C::AbstractMatrix{F})::Int64 where {F <: Real}
    num_points = size(C, 1)
    max_p::Int64 = 0
    for i in 1:num_points
        for j in 1:(i - 1)
            if C[i, j] != 0.0
                max_p = max(max_p, abs(i - j))
            end
        end
    end
    return max_p
end

"""
Compute Banded Cholesky Factorization using Algorithm 2.9 in the book "Gaussian Markov Random Fields: Theory and Applications" by H. Rue and L. Held (2005).

"""

function compute_banded_cholesky!(
    L::AbstractMatrix{F},
    Q::AbstractMatrix{F},
    p::Int64
) where {F <: Real}

    n = size(L, 1)
    v = zeros(F, n)

    @inbounds for j in 1:n  # @inbounds disabilita i controlli sui limiti (più veloce)
        v .= 0.0            # Azzera il vettore sul posto
        lambda = min(j + p, n)
        
        # Usiamo @views per non allocare e .= per assegnare sul posto
        @views v[j:lambda] .= Q[j:lambda, j]
        lb = max(1, j - p)
        
        for k in lb:(j - 1)
            i = min(k + p, n)
            # Operazioni vettoriali fuse senza allocazioni
            @views v[j:i] .-= L[j:i, k] .* L[j, k]
        end
        
        @views L[j:lambda, j] .= v[j:lambda] ./ sqrt(v[j])
    end
end

"""
Compute the adjancency matrix for a set of points using DelaunayTriangulation.jl.
"""
function compute_adjacency_matrix(points::AbstractMatrix{F}) where {F <: Real}
    num_points = size(points, 1)
    
    # 1. Triangolazione (richiede matrice 2xN, quindi trasponiamo)
    tri = triangulate(points')
    
    # 2. Estrazione degli spigoli per costruire la matrice sparsa
    # Pre-allochiamo assumendo mediamente 6 vicini per nodo in una mesh 2D
    I = Int[]
    J = Int[]
    sizehint!(I, num_points * 7)
    sizehint!(J, num_points * 7)
    
    for e in each_edge(tri)
        u, v = DelaunayTriangulation.initial(e), DelaunayTriangulation.terminal(e)
        # Ignora l'orizzonte (Boundary nodes di DelaunayTriangulation)
        if u > 0 && v > 0 && u <= num_points && v <= num_points
            push!(I, u); push!(J, v)
            push!(I, v); push!(J, u) # Grafo non diretto (simmetrico)
        end
    end
    
    # Aggiungiamo la diagonale
    for i in 1:num_points
        push!(I, i); push!(J, i)
    end
    
    V = ones(Int, length(I))
    
    adjacency_matrix = sparse(I, J, V, num_points, num_points)
    return adjacency_matrix
end


function find_optimal_permutation(
    A::AbstractMatrix{F}
) where {F <: Real}
    permutation = symrcm(A)
    A .= A[permutation, permutation]
    return A, permutation
end

"""
Using Theorem 2.8 and Corollary 2.2 in "Gaussian Markov Random Fields: Theory and Applications" by H. Rue and L. Held (2005) to check if given two points i < j the element L(j, i) is zero. The parents of the node i will be the non-null elements of the i-th row of L.
"""

function symbolic_cholesky_parents(A::SparseMatrixCSC)::Vector{Vector{Int}}
    n = size(A, 1)
    adj = [Set{Int}() for _ in 1:n]
    rows = rowvals(A)
    for i in 1:n
        for idx in nzrange(A, i)
            j = rows[idx]
            if j > i
                push!(adj[i], j)
            end
        end
    end
    parents_list = [Int[] for _ in 1:n]
    for i in 1:n
        neighbors = sort!(collect(adj[i]))
        
        # Tutti questi vicini avranno 'i' come genitore nel fattore di Cholesky
        for j in neighbors
            push!(parents_list[j], i)
        end
        if length(neighbors) > 1
            nxt = neighbors[1] # Il prossimo nodo che verrà eliminato in questo gruppo
            for k in 2:length(neighbors)
                push!(adj[nxt], neighbors[k]) # Aggiungiamo l'arco di fill-in
            end
        end
    end
    return parents_list
end
