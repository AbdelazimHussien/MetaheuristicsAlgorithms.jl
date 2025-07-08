@testset "AHA" verbose = true begin 
    @warn "AHA tests are failing, please check the implementation."
    @testset "Ackley with AHA, p = 2" verbose = true begin 
        lb = [-32.768 for i in 1:2]
        ub = [32.768 for i in 1:2]
        result = AHA(Ackley, lb, ub, 100, 1500)#AHA(100, 1500, lb, ub, Ackley)
        @test isapprox(result.BestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end 
    @testset "Ackley with AHA, p = 5" verbose = true begin 
        lb = [-32.768 for i in 1:5]
        ub = [32.768 for i in 1:5]
        result = AHA(Ackley, lb, ub, 100, 1500)#AHA(100, 1500, lb, ub, Ackley)
        @test isapprox(result.BestX, [0.0 for i in 1:5], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with AHA, p = 2" verbose = true begin 
        lb = [-600 for i in 1:2]
        ub = [600 for i in 1:2]
        result = AHA(Griewank, lb, ub, 100, 1500)#AHA(100, 1500, lb, ub, Griewank)
        @test isapprox(result.BestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end
end 