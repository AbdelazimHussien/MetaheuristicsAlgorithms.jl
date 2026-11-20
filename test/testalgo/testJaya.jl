@testset "Jaya" verbose = true begin
    @testset "Ackley with Jaya, p = 2" verbose = true begin
        Random.seed!(1)
        lb = [-32.768 for i = 1:2]
        ub = [32.768 for i = 1:2]
        result = Jaya(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Ackley with Jaya, p = 5" verbose = true begin
        Random.seed!(1)
        lb = [-32.768 for i = 1:5]
        ub = [32.768 for i = 1:5]
        result = Jaya(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:5], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with Jaya, p = 2" verbose = true begin
        Random.seed!(1)
        lb = [-600.0 for i = 1:2]
        ub = [600.0 for i = 1:2]
        result = Jaya(Griewank, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=0.5)
        @test isapprox(result.bestF, 0.0, atol=0.005)
    end
end
