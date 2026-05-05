module GaussianMarkovRandomFields

using Random
using Distributions
using LinearAlgebra
using SparseArrays
using StatsFuns
using SpecialFunctions
using DelaunayTriangulation
using SymRCM
using FFTW

abstract type AbstractMesh end

include("utils.jl")

include("circulant_gaussian_markov_random_field_1d.jl")
export CirculantGaussianMarkovRandomField1D

include("strategy.jl")
export MaximinOrderingStrategy, DelaunayOrderingStrategy

include("nngp.jl")
export NearestNeighbourGaussianProcess

end # module GaussianMarkovRandomFields
