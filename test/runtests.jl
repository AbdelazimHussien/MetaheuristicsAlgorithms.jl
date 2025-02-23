using Pkg
Pkg.activate(".")
using MetaheuristicsAlgorithms
using Test
using Random

Random.seed!(1)
include("benchmark.jl")

@testset "MetaheuristicsAlgorithms.jl" begin
    @testset "AEFA on EGGHOLDER FUNCTION" begin
        lb = [-512, -512]
        ub = [512, 512]
        algo = AEFA(100, 300, lb, ub)

        while algo.update
            y = benchmark(algo.x; funNo=1)
            update_state!(algo, y)
        end
        @test algo.gbest == [-461.648054120324, -390.06329199871277]
        @test algo.gbest_val[end] == -775.8393103361666
    end
end
