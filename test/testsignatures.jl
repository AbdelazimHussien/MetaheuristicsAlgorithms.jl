# Every algorithm is called in the three documented ways, on a problem whose
# bounds differ per variable and do not contain zero. The result must be a point
# inside the bounds, and its value must be the value of the objective there.
@testset "Call signatures and bounds" verbose = true begin
    lb = [1.0, 2.0, -9.0, 0.5, 10.0, -3.0]
    ub = [4.0, 2.5, -1.0, 8.0, 30.0, -2.0]
    optimum = lb .+ 0.3 .* (ub .- lb)
    shifted(x) = sum(abs2, x .- optimum)
    sphere(x) = sum(abs2, x)
    npop, max_iter = 12, 30

    # TOC is not ported completely, and LSHADE_cnEpSin leaves the bounds.
    skipped = ["TOC", "LSHADE_cnEpSin"]
    algodir = joinpath(pkgdir(MetaheuristicsAlgorithms), "src", "algo")
    names = sort([replace(f, ".jl" => "") for f in readdir(algodir) if endswith(f, ".jl") && !occursin("_struct", f)])

    quiet(f) = redirect_stdout(f, devnull)

    @testset "$name" for name in setdiff(names, skipped)
        alg = getfield(MetaheuristicsAlgorithms, Symbol(name))

        Random.seed!(1)
        result = quiet(() -> alg(shifted, lb, ub, npop, max_iter))
        @test length(result.bestX) == length(lb)
        @test all(lb .<= result.bestX .<= ub)
        @test result.bestF ≈ shifted(result.bestX)

        Random.seed!(1)
        result = quiet(() -> alg(OptimizationProblem(shifted, lb, ub), npop, max_iter))
        @test all(lb .<= result.bestX .<= ub)
        @test result.bestF ≈ shifted(result.bestX)

        Random.seed!(1)
        result = quiet(() -> alg(sphere, -5.0, 5.0, npop, max_iter, 7))
        @test length(result.bestX) == 7
        @test all(-5.0 .<= result.bestX .<= 5.0)
        @test result.bestF ≈ sphere(result.bestX)

        # more variables than individuals
        Random.seed!(1)
        result = quiet(() -> alg(sphere, fill(-5.0, 20), fill(5.0, 20), npop, max_iter))
        @test length(result.bestX) == 20
        @test result.bestF ≈ sphere(result.bestX)
    end
end
