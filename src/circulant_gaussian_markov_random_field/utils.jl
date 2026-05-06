"""
Compute the eigenvalues for a circulant matrix with coefficient delta and offset.

"""
function _compute_circulant_eigenvalues_1d(delta::F, n::Int; offset::F = 2.0)::Vector{F} where {F <: Real}
    eigenvalues = zeros(F, n)
    c0 = delta + offset
    for j in 1:n
        j_prime = j - 1
        eigenvalues[j] = c0 - 2 * cos(2 * pi * j_prime / n)
    end

    return eigenvalues
end

function _compute_circulant_precision_matrix_1d!(
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
