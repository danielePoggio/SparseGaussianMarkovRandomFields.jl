using Test
using GaussianMarkovRandomFields

@testset "GaussianMarkovRandomFields.jl" begin
    # Test di base: verifichiamo che la struct si inizializzi correttamente
    dist = CirculantGaussianMarkovRandomField1D(10, 1.5)
    
    @test length(dist) == 10
    @test eltype(dist) == Float64
end