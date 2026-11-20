@testset "ParrotO" verbose = true begin
    @testset "Ackley with ParrotO, p = 2" verbose = true begin
        Random.seed!(1)
        lb = [-32.768 for i = 1:2]
        ub = [32.768 for i = 1:2]
        result = ParrotO(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=0.0001)
        @test isapprox(result.bestF, 0.0, atol=0.0001)
    end
    @testset "Ackley with ParrotO, p = 5" verbose = true begin
        Random.seed!(1)
        lb = [-32.768 for i = 1:5]
        ub = [32.768 for i = 1:5]
        result = ParrotO(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:5], atol=0.0005)
        @test isapprox(result.bestF, 0.0, atol=0.0005)
    end
    @testset "Griewank with ParrotO, p = 2" verbose = true begin
        Random.seed!(1)
        lb = [-600.0 for i = 1:2]
        ub = [600.0 for i = 1:2]
        result = ParrotO(Griewank, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=0.0001)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
end
