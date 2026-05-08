module SparseGaussianMarkovRandomFields
export SparseGaussianMarkovRandomFields

using Random
using Distributions
using LinearAlgebra
using SparseArrays
using StatsFuns
using SpecialFunctions
using DelaunayTriangulation
using SymRCM
using FFTW


"""
The scope of this object is to store the informations related to the structure of the precision precision and preallocate it, in order to speed up the computation for MCMC algorithms.
"""
abstract type AbstractCache end


"""
This is a general interface for every Gaussian Markov Random Field object with some fallback methods.
"""

abstract type AbstractGaussianMarkovRandomField <: Distributions.ContinuousMultivariateDistribution end

Base.length(dist::AbstractGaussianMarkovRandomField) = dist.n
Distributions.mean(d::AbstractGaussianMarkovRandomField) = zeros(length(d))

function invcov!(Q::AbstractMatrix{F}, dist::AbstractGaussianMarkovRandomField) where {F <: Real} end

function Distributions._logpdf(dist::AbstractGaussianMarkovRandomField, x::AbstractVector{F}) where {F <: Real} end

function logpdf_nograd(dist::AbstractGaussianMarkovRandomField, x::AbstractVector{F}) where {F <: Real} end


export AbstractGaussianMarkovRandomField, invcov!, logpdf_nograd

include("circulant_gaussian_markov_random_field/CirculantGaussianMarkovRandomField.jl")

using .CirculantGaussianMarkovRandomField
import .CirculantGaussianMarkovRandomField: CirculantGaussianMarkovRandomField1D
export CirculantGaussianMarkovRandomField1D

include("nearest_neighbours_gaussian_process/NNGP.jl")

using .NNGP
import .NNGP: NearestNeighbourGaussianProcess, MaximinOrderingStrategy, update_precision_values!
export NearestNeighbourGaussianProcess, MaximinOrderingStrategy, update_precision_values!

# include("spde/SPDE.jl")

# using .SPDE
# import .SPDE: SPDEMesh, SPDEMatern
# export SPDEMesh, SPDEMatern

end # module SparseGaussianMarkovRandomFields
