module CirculantGaussianMarkovRandomField

using Random
using Distributions
using LinearAlgebra
using SparseArrays
using StatsFuns
using SpecialFunctions
using FFTW

import ..GaussianMarkovRandomFields: AbstractGaussianMarkovRandomField, AbstractCache

include("utils.jl")

include("circulant_gaussian_markov_random_field_1d.jl")

export CirculantGaussianMarkovRandomField1D, invcov


end
