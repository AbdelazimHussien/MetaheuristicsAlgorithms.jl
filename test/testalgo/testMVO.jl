@testset "MVO" verbose = true begin
    @testset "Ackley with MVO, p = 2" verbose = true begin
        Random.seed!(1)
        lb = [-32.768 for i = 1:2]
        ub = [32.768 for i = 1:2]
        result = MVO(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=0.005)
        @test isapprox(result.bestF, 0.0, atol=0.01)
    end
    @testset "Ackley with MVO, p = 5" verbose = true begin
        Random.seed!(1)
        lb = [-32.768 for i = 1:5]
        ub = [32.768 for i = 1:5]
        result = MVO(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:5], atol=0.05)
        @test isapprox(result.bestF, 0.0, atol=0.05)
    end
    @testset "Griewank with MVO, p = 2" verbose = true begin
        Random.seed!(1)
        lb = [-600.0 for i = 1:2]
        ub = [600.0 for i = 1:2]
        result = MVO(Griewank, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=0.1)
        @test isapprox(result.bestF, 0.0, atol=0.0005)
    end
end
