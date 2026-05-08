using Test
using ADTypes
using GaussianMarkovRandomFields
using ForwardDiff
using Optim
using Random
using Distributions

# Disattiviamo log troppo verbosi durante i test
using Logging
disable_logging(Logging.Info)

@testset "GaussianMarkovRandomFields.jl" begin

    @testset "NNGP - Topologia e Costruttori" begin
        # Setup dati
        points = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
        n_neighs = 2
        
        # Test 1: Creazione Strategy
        strategy = MaximinOrderingStrategy(points, n_neighs)
        @test size(strategy.points, 1) == 4
        @test length(strategy.neighbors) == 4
        
        # Test 2: Costruttore Float64 (Alloca e calcola D e V)
        dist = NearestNeighbourGaussianProcess(strategy, 1.0, 1.5)
        @test dist.n == 4
        @test dist.variance == 1.0
        @test length(dist.strategy.D) == 4
        @test dist.strategy.D[1] == 1.0 # La prima varianza condizionale è pari alla varianza marginale
        
        # Test 3: Costruttore Dual (Salta il calcolo, array di zeri)
        dual_var = ForwardDiff.Dual(1.0, 1.0)
        dist_dual = NearestNeighbourGaussianProcess(strategy, dual_var, dual_var)
        @test dist_dual.strategy.D[1] == 1.0 # Perché nel costruttore Dual non calcoliamo nulla!
    end

    @testset "NNGP - Automatic Differentiation" begin
        points = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
        strategy = MaximinOrderingStrategy(points, 2)
        y = [0.1, -0.2, 0.3, -0.1]
        
        # Funzione target per ForwardDiff
        function test_logpdf(params)
            d = NearestNeighbourGaussianProcess(strategy, params[1], params[2])
            return logpdf(d, y)
        end
        
        # Test 4: ForwardDiff non deve crashare e deve restituire un gradiente di taglia 2
        grad = ForwardDiff.gradient(test_logpdf, [1.0, 1.5])
        @test length(grad) == 2
        @test !isnan(grad[1]) # Il gradiente non deve essere NaN
        @test !isnan(grad[2])
    end

    @testset "NNGP - Inferenza MLE" begin
        # Generiamo dati veri e testiamo se Optim ritrova (approssimativamente) i parametri
        nx, ny = 10, 10
        x_seq = range(0, 1, length=nx+1)
        y_seq = range(0, 1, length=ny+1)
        points = zeros((nx+1)*(ny+1), 2)
        points[:, 1] = repeat(x_seq, inner = ny + 1)
        points[:, 2] = repeat(y_seq, outer = nx + 1)
        distance_min = x_seq[2] - x_seq[1]
        distance_max = sqrt(2)
        rho_min = 3.0 / distance_max
        rho_max = 3.0 / distance_min

        rho = 6.0
        variance = 1.0
        strategy = MaximinOrderingStrategy(points, 10)
        rng = MersenneTwister(42)
        dist_vera = NearestNeighbourGaussianProcess(strategy, variance, rho)
        y = zeros((nx+1)*(ny+1))
        rand!(rng, dist_vera, y)
        
        function neg_log_likelihood(p)
            d = NearestNeighbourGaussianProcess(strategy, exp(p[1]), exp(p[2]))
            return -logpdf(d, y)
        end
        
        initial_guess = [log(0.5), log(0.5 * (rho_max + rho_min))]
        res = optimize(neg_log_likelihood, [-5.0, log(rho_min)], [5.0, log(rho_max)], initial_guess, Fminbox(LBFGS()); autodiff=AutoForwardDiff())
        
        var_est, rho_est = exp.(Optim.minimizer(res))
        
        # Test 5: Verifica convergenza entro tolleranze
        # Non possiamo pretendere uguaglianza perfetta in statistica, usiamo isapprox (o atol/rtol)
        @test isapprox(abs(var_est- variance) / variance, 0.0, atol=1.0)
        @test isapprox(abs(rho_est - rho) / rho, 0.0, atol=1.0)
        @test Optim.converged(res) == true
    end

    # Puoi aggiungere un ulteriore blocco per il Modello Circolare...
    @testset "Circulant GMRF 1D" begin
        dist = CirculantGaussianMarkovRandomField1D(10, 1.5)
        
        @test length(dist) == 10
        @test eltype(dist) == Float64
    end
end