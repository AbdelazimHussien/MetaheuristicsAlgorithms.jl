@testset "AFT" verbose = true begin 
    @testset "Ackley with AFT, p = 5" verbose = true begin 
        lb = [-32.768 for i in 1:5]
        ub = [32.768 for i in 1:5]
        result = AFT(Ackley, lb, ub, 100, 500)
        @test isapprox(result.bestX, [0.0 for i in 1:5], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end 
    @testset "Ackley with AFT, p = 10" verbose = true begin 
        lb = [-32.768 for i in 1:10]
        ub = [32.768 for i in 1:10]
        result = AFT(Ackley, lb, ub, 100, 500)#AFT(5, 5000, lb, ub, Ackley)
        @test isapprox(result.bestX, [0.0 for i in 1:10], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with AFT, p = 2" verbose = true begin 
        lb = [-600.0 for i in 1:2]
        ub = [600.0 for i in 1:2]
        result = AFT(Griewank, lb, ub, 100, 500)#AFT(2, 5000, lb, ub, Griewank)
        @test isapprox(result.bestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
end 