@testset "AEO" verbose = true begin
    @testset "Ackley with AEO, p = 2" verbose = true begin
        lb = [-32.768 for i = 1:2]
        ub = [32.768 for i = 1:2]
        result = AEO(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Ackley with AEO, p = 5" verbose = true begin
        lb = [-32.768 for i = 1:5]
        ub = [32.768 for i = 1:5]
        result = AEO(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:5], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with AEO, p = 2" verbose = true begin
        lb = [-600.0 for i = 1:2]
        ub = [600.0 for i = 1:2]
        result = AEO(Griewank, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Eggholder with AEO, p = 2, fixed seed" verbose = true begin
        Random.seed!(1)
        lb = fill(-512.0, 2)
        ub = fill(512.0, 2)
        result = AEO(eggholder, lb, ub, 100, 1000)
        @test result.bestX ≈ [512.0, 404.23180515536245]
        @test result.bestF ≈ -959.640662720851
    end
end

# using Profile, PProf
# Profile.clear()
# @pprof for _ = 1:100
#     AEO(eggholder, lb, ub, 100, 1000)
# end

# Profile.Allocs.clear()
# Profile.Allocs.@profile AEO(eggholder, lb, ub, 100, 1000)
# PProf.Allocs.pprof(out="profile.pb.gz", webport=62262)