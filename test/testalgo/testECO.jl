@testset "ECO" verbose = true begin 
    @testset "Ackley with ECO, p = 2" verbose = true begin 
        lb = [-32.768 for i in 1:2]
        ub = [32.768 for i in 1:2]
        result = ECO(Ackley, lb, ub, 100, 500)#DSO(100, 500, lb, ub, Ackley)
        @test isapprox(result.bestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end 
    @testset "Ackley with ECO, p = 5" verbose = true begin 
        lb = [-32.768 for i in 1:5]
        ub = [32.768 for i in 1:5]
        result = ECO(Ackley, lb, ub, 100, 500)#DSO(100, 500, lb, ub, Ackley)
        @test isapprox(result.bestX, [0.0 for i in 1:5], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with ECO, p = 2" verbose = true begin 
        lb = [-600.0 for i in 1:2]
        ub = [600.0 for i in 1:2]
        result = ECO(Griewank, lb, ub, 100, 500)#DSO(100, 500, lb, ub, Griewank)
        @test isapprox(result.bestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
end 
# DSO(Ackley, lb, ub, 100, 500)