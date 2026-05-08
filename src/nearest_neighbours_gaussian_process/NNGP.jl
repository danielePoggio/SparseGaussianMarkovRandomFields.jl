module NNGP
export NNGP

using Random
using Distributions
using LinearAlgebra
using SparseArrays
import ..GaussianMarkovRandomFields: AbstractGaussianMarkovRandomField, AbstractCache

include("strategy.jl")
include("utils.jl")
include("nnpg.jl")

export NearestNeighbourGaussianProcess, MaximinOrderingStrategy, update_precision_values!

end