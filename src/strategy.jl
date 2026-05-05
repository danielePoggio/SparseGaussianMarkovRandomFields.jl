using Base.Sort

abstract type AbstractStrategy end

# ----------------------------------------------------------------
# MinMax Ordering Strategy
# ----------------------------------------------------------------

struct MaximinOrderingStrategy{F <: Real} <: AbstractStrategy
    num_neighbors::Int
    points::Matrix{F}
    neighbors::Vector{Vector{Int}}
    permutation::Vector{Int}
    inv_permutation::Vector{Int}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{F}
    D::Vector{F}
end

function MaximinOrderingStrategy(
    points::Matrix{F},
    num_neighbors::Int
) where {F <: Real}

    num_points = size(points, 1) # Assumiamo dimensione N x D
    permutation = zeros(Int, num_points)
    unselected = trues(num_points)
    min_dists = fill(F(Inf), num_points) # Tipizzato con F per stabilità

    curr_idx = 1
    permutation[1] = curr_idx
    unselected[curr_idx] = false
    
    for i in 2:num_points
        best_dist = -one(F)
        next_idx = -1
        curr_pt = @view points[curr_idx, :]
        
        for j in 1:num_points
            if unselected[j]
                d = sum(abs2, @view(points[j, :]) .- curr_pt)
                
                # Fase "Min"
                if d < min_dists[j]
                    min_dists[j] = d
                end
                
                # Fase "Max"
                if min_dists[j] > best_dist
                    best_dist = min_dists[j]
                    next_idx = j
                end
            end
        end
        
        curr_idx = next_idx
        permutation[i] = curr_idx
        unselected[curr_idx] = false
    end

    inv_permutation = zeros(Int, num_points)
    for i in 1:num_points
        inv_permutation[permutation[i]] = i
    end

    neighbors = Vector{Vector{Int}}(undef, num_points)
    neighbors[1] = Int[]
    
    for i in 2:num_points
        k = min(i - 1, num_neighbors)
        target_pt = @view points[permutation[i], :]
        dists = zeros(F, i - 1)

        for j in 1:(i - 1)
            prev_pt = @view points[permutation[j], :]
            dists[j] = sum(abs2, prev_pt .- target_pt)
        end
        
        closest_indices = partialsortperm(dists, 1:k)
        neighbors[i] = collect(closest_indices)
    end

    # Creazione efficiente della matrice sparsa
    I = Int[]
    J = Int[]
    sizehint!(I, num_points * num_neighbors)
    sizehint!(J, num_points * num_neighbors)
    
    for i in 2:num_points
        k = min(i - 1, num_neighbors)
        for j in 1:k
            push!(I, i); push!(J, neighbors[i][j])
        end
    end

    V = ones(length(I))
    D = zeros(F, num_points)

    points_permuted = points[permutation, :]

    return MaximinOrderingStrategy{F}(
        num_neighbors,
        points_permuted,
        neighbors,
        permutation,
        inv_permutation,
        I,
        J,
        V,
        D
    )
end

# ----------------------------------------------------------------
# Delaunay Ordering Strategy
# ----------------------------------------------------------------

struct DelaunayOrderingStrategy{F <: Real} <: AbstractStrategy
    num_degree::Int
    num_neighbors::Int
    points::Matrix{F}
    neighbors::Vector{Vector{Int}}
    permutation::Vector{Int}
    inv_permutation::Vector{Int}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{F}
    D::Vector{F}
end

function DelaunayOrderingStrategy(
    points::Matrix{F},
    num_degree::Int
) where {F <: Real}
    num_points = size(points, 1)
    adjacency_matrix = compute_adjacency_matrix(points)
    adjacency_degree_matrix = adjacency_matrix^(num_degree)
    A, permutation = find_optimal_permutation(adjacency_degree_matrix)
    inv_permutation = zeros(Int, num_points)
    for i in 1:num_points
        inv_permutation[permutation[i]] = i
    end

    # neighbors = symbolic_cholesky_parents(adjacency_degree_matrix)
    max_bandwidth = find_bandwidth(A)
    neighbors = [Int[] for _ in 1:num_points]
    for i in 2:num_points
        k = min(i - 1, max_bandwidth)
        neighbors[i] = collect((i - k):(i - 1))
    end

    # Creazione efficiente della matrice sparsa
    I = Int[]
    J = Int[]
    sizehint!(I, num_points * max_bandwidth)
    sizehint!(J, num_points * max_bandwidth)
    for i in 2:num_points
        num_neighs_i = length(neighbors[i])
        for j in 1:num_neighs_i
            push!(I, i); push!(J, neighbors[i][j])
        end
    end
    V = ones(length(I))
    D = zeros(F, num_points)

    points_permuted = points[permutation, :]

    return DelaunayOrderingStrategy{F}(
        num_degree,
        max_bandwidth,
        points_permuted,
        neighbors,
        permutation,
        inv_permutation,
        I,
        J,
        V,
        D
    )
end