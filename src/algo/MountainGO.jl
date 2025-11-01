
function Coefficient_Vector(dim, Iter, max_iter)
    a2 = -1 + Iter * (-1 / max_iter)
    u = randn(dim)
    v = randn(dim)

    cofi = zeros(Float64, 4, dim)

    cofi[1, :] .= rand(dim)
    cofi[2, :] .= (a2 + 1) .+ rand(dim)
    cofi[3, :] .= a2 .* randn(dim)
    cofi[4, :] .= u .* (v .^ 2) .* cos.(rand() * 2 .* u)  # Fill the fourth row

    return cofi
end

function Solution_Imp(X, BestX, lb, ub, npop, cofi, M, A, D, i)
    NewX = zeros(Float64, 4, size(X, 2))
    NewX[1, :] .= (ub .- lb) .* rand(size(X, 2)) .+ lb
    NewX[2, :] .= BestX .- abs.((rand(1:2) .* M .- rand(1:2) .* X[i, :]) .* A) .* cofi[rand(1:4), :]  # Second row
    NewX[3, :] .= (M .+ cofi[rand(1:4), :]) .+ (rand(1:2) .* BestX .- rand(1:2) .* X[rand(1:npop), :]) .* cofi[rand(1:4), :]  # Third row
    NewX[4, :] .= (X[i, :] .- D) .+ (rand(1:2) .* BestX .- rand(1:2) .* M) .* cofi[rand(1:4), :]  # Fourth row

    return NewX
end

"""
# References:

- Abdollahzadeh, Benyamin, Farhad Soleimanian Gharehchopogh, Nima Khodadadi, and Seyedali Mirjalili.
  "Mountain gazelle optimizer: a new nature-inspired metaheuristic algorithm for global optimization problems."
  Advances in Engineering Software 174 (2022): 103282.
"""
function MountainGO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return MountainGO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function MountainGO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    lb = ones(dim) .* lb    # Lower Bound
    ub = ones(dim) .* ub    # Upper Bound

    X = initialization(npop, dim, ub, lb)

    BestX = []
    BestFitness = Inf
    BestF = 0.0

    Sol_Cost = zeros(npop)
    for i = 1:npop
        Sol_Cost[i] = objfun(X[i, :])
        if Sol_Cost[i] <= BestFitness
            BestFitness = Sol_Cost[i]
            BestX = X[i, :]
        end
    end

    cnvg = zeros(max_iter)
    for Iter = 1:max_iter
        for i = 1:npop
            RandomSolution = rand(1:npop, ceil(Int, npop / 3))
            M = X[rand(ceil(Int, npop / 3):npop), :] * floor(rand()) + reshape(mean(X[RandomSolution, :], dims=1), 30) .* ceil(rand())

            cofi = Coefficient_Vector(dim, Iter, max_iter)

            A = randn(dim) .* exp(2 - Iter * (2 / max_iter))
            D = (abs.(X[i, :]) .+ abs.(BestX)) .* (2 * rand() .- 1)

            NewX = Solution_Imp(X, BestX, lb, ub, npop, cofi, M, A, D, i)

            NewX .= max.(lb, min.(NewX, ub))
            Sol_CostNew = [objfun(row) for row in eachrow(NewX)]

            X = vcat(X, NewX)
            Sol_Cost = vcat(Sol_Cost, Sol_CostNew)

            _, idbest = findmin(Sol_Cost)
            BestX = X[idbest, :]
        end

        SortOrder = sortperm(Sol_Cost)
        Sol_Cost = Sol_Cost[SortOrder]

        X = X[SortOrder, :]
        BestFitness, idbest = findmin(Sol_Cost)
        BestX = X[idbest, :]
        X = X[1:npop, :]
        Sol_Cost = Sol_Cost[1:npop]
        cnvg[Iter] = BestFitness
        BestF = BestFitness
    end

    # return BestF, BestX, cnvg
    return OptimizationResult(
        BestX,
        BestF,
        cnvg)
end

function MountainGO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return MountainGO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end
