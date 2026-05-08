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
    V::AbstractVector{T},
    D::AbstractVector{T},
    neighbors::Vector{Vector{Int}},
    points::AbstractMatrix{F},
    variance::T,
    rho::T
) where {T <: Real, F <: Real}
    
    max_neighs = maximum(length.(neighbors))
    # Allochiamo le cache interne col tipo T (Dual) per tracciare i gradienti
    C_corr_cache = zeros(T, max_neighs, max_neighs)
    C_vec_cache = zeros(T, max_neighs)
    A_cache = zeros(T, max_neighs)
    num_points = size(points, 1)

    v_idx = 1
    D[1] = variance
    for i in 2:num_points
        C_corr_cache .= zero(T)
        C_vec_cache .= zero(T)
        A_cache .= zero(T)
        
        Ni = neighbors[i]
        num_neighs_i = length(Ni)
        
        C_mat = view(C_corr_cache, 1:num_neighs_i, 1:num_neighs_i)
        c_vec = view(C_vec_cache, 1:num_neighs_i)
        a_vec = view(A_cache, 1:num_neighs_i)
        
        for j in 1:num_neighs_i
            idxj = Ni[j]
            dij = _fast_distance_computation(points, i, idxj) # Restituisce F (Float64)
            
            # variance (Dual) * exp( Dual * Float64 ) = Dual! 
            # I tipi si fondono magicamente e le derivate si salvano!
            c_vec[j] = variance * exp(- rho * dij)
            C_mat[j, j] = variance + 1e-8
            
            for k in (j + 1):num_neighs_i
                idxk = Ni[k]
                djk = _fast_distance_computation(points, idxj, idxk)
                C_mat[j, k] = variance * exp(- rho * djk)
                C_mat[k, j] = C_mat[j, k]
            end
        end

        a_vec .= c_vec
        chol_C = cholesky!(Symmetric(C_mat))
        ldiv!(chol_C, a_vec)
        D[i] = variance - dot(a_vec, c_vec)
        
        for j in 1:num_neighs_i
            V[v_idx] = a_vec[j]
            v_idx += 1
        end
    end
end

function _quadratic_form(
    x::AbstractVector{T}, # T: tipo dei Dati (es. Float64)
    neighbors::Vector{Vector{Int}},
    V::AbstractVector{F}, # F: tipo dei Parametri (es. Dual)
    D::AbstractVector{F}  # F: tipo dei Parametri (es. Dual)
) where {T <: Real, F <: Real}
    
    num_points = length(x)
    
    # Troviamo il tipo "vincitore" tra Float64 e Dual (sarà Dual)
    ResultType = promote_type(T, F)
    q = zero(ResultType) 
    
    v_idx = 1
    
    @inbounds for i in 1:num_points
        # Inizializziamo u_i col tipo flessibile per evitare errori
        u_i = ResultType(x[i])
        
        Ni = neighbors[i]
        num_neighs_i = length(Ni)
        
        for j in 1:num_neighs_i
            col = Ni[j]
            val = V[v_idx]
            
            u_i -= val * x[col]
            v_idx += 1
        end
        
        q += (u_i^2) / D[i]
    end
    
    return q
end

function _build_precision_coordinates( 
    neighbors::Vector{Vector{Int}}
)
    n = length(neighbors)
    num_max_neighs = maximum(length.(neighbors))
    estimated_len = n * num_max_neighs 
    
    I_Q = Int[]
    J_Q = Int[]
    sizehint!(I_Q, estimated_len)
    sizehint!(J_Q, estimated_len)
    
    for i in 1:n
        # 1. Coordinata Diagonale
        push!(I_Q, i); push!(J_Q, i)
        
        Ni = neighbors[i]
        num_neighs = length(Ni)
        
        for j in 1:num_neighs
            idx_j = Ni[j]
            
            # 2. Coordinate Nodo-Vicino (Simmetrico)
            push!(I_Q, i); push!(J_Q, idx_j)
            push!(I_Q, idx_j); push!(J_Q, i)
            
            # Coordinata per il contributo alla diagonale del vicino
            push!(I_Q, idx_j); push!(J_Q, idx_j)
            
            # 3. Coordinate Moralizzazione (Vicino-Vicino)
            for k in (j+1):num_neighs
                idx_k = Ni[k]
                push!(I_Q, idx_j); push!(J_Q, idx_k)
                push!(I_Q, idx_k); push!(J_Q, idx_j)
            end
        end
    end
    
    return I_Q, J_Q
end

function _compute_precision_values!(
    V_Q::AbstractVector{F}, 
    V::AbstractVector{F}, 
    D::AbstractVector{F}, 
    neighbors::Vector{Vector{Int}}
) where {F <: Real}
    
    n = length(D)
    v_input_idx = 1  
    vq_output_idx = 1 
    
    @inbounds for i in 1:n
        inv_Di = 1.0 / D[i]
        
        # 1. Valore Diagonale
        V_Q[vq_output_idx] = inv_Di
        vq_output_idx += 1
        
        num_neighs = length(neighbors[i])
        
        for j in 1:num_neighs
            v_j = V[v_input_idx + j - 1]
            
            # 2. Valori Nodo-Vicino
            val_nv = -v_j * inv_Di
            V_Q[vq_output_idx] = val_nv; vq_output_idx += 1
            V_Q[vq_output_idx] = val_nv; vq_output_idx += 1
            
            # Contributo diagonale vicino
            V_Q[vq_output_idx] = (v_j^2) * inv_Di
            vq_output_idx += 1
            
            # 3. Valori Moralizzazione
            for k in (j+1):num_neighs
                v_k = V[v_input_idx + k - 1]
                
                val_moral = (v_j * v_k) * inv_Di
                V_Q[vq_output_idx] = val_moral; vq_output_idx += 1
                V_Q[vq_output_idx] = val_moral; vq_output_idx += 1
            end
        end
        v_input_idx += num_neighs
    end
    
    return V_Q
end
