module GaussianMarkovRandomFields
export GaussianMarkovRandomFields

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

include("circulant_gaussian_markov_random_field/CirculantGaussianMarkovRandomField.jl")

using .CirculantGaussianMarkovRandomField
import .CirculantGaussianMarkovRandomField: CirculantGaussianMarkovRandomField1D
export CirculantGaussianMarkovRandomField1D

include("nearest_neighbours_gaussian_process/NNGP.jl")

using .NNGP
import .NNGP: NearestNeighbourGaussianProcess, MaximinOrderingStrategy
export NearestNeighbourGaussianProcess, MaximinOrderingStrategy

include("spde/SPDE.jl")

using .SPDE
import .SPDE: SPDEMesh, SPDEMatern
export SPDEMesh, SPDEMatern

end # module GaussianMarkovRandomFields
