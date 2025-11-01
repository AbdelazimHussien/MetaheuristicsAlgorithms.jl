"""
    AEO(npop, max_iter, lb::Real, ub::Real, dim, objfun)
    AEO(npop, max_iter, lb::Vector{Float64}, ub::Vector{Float64}, objfun)
    AEO(problem::OptimizationProblem, npop=30, max_iter=1000)

Run the Artificial Ecosystem-based Optimization (AEO) algorithm.

This function supports three ways to define the optimization problem:

1. By passing scalar bounds and problem dimensionality.
2. By passing vector bounds explicitly.
3. By using an `OptimizationProblem` struct that encapsulates the problem definition.

# Arguments

## Common
- `npop::Integer`: Number of individuals in the population.
- `max_iter::Integer`: Maximum number of iterations.

## For scalar bounds:
- `lb::Real`: Lower bound (same for all dimensions).
- `ub::Real`: Upper bound (same for all dimensions).
- `dim::Integer`: Dimensionality of the problem.
- `objfun::Function`: Objective function to minimize.

## For vector bounds:
- `lb::Vector{Float64}`: Lower bounds for each dimension.
- `ub::Vector{Float64}`: Upper bounds for each dimension.
- `objfun::Function`: Objective function to minimize.

## For `OptimizationProblem` form:
- `problem::OptimizationProblem`: A struct containing `objfun`, `lb`, `ub`, and `dim`.

# Returns

- `OptimizationResult`: A struct containing:
  - `bestX::Vector{Float64}`: The best solution found.
  - `bestF::Float64`: The best objective value.
  - `his_best_fit::Vector{Float64}`: History of the best fitness value at each iteration.

# Examples

```julia
# Using scalar bounds and dimension
result = AEO(30, 100, -5.12, 5.12, 10, Ackley)

# Using vector bounds
lb = fill(-5.12, 10)
ub = fill(5.12, 10)
result = AEO(30, 100, lb, ub, 10, Ackley)

# Using OptimizationProblem struct
problem = OptimizationProblem(Ackley, -5.12, 5.12, 10)
result = AEO(problem, 30, 100)
"""
function AEO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return AEO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function AEO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    PopPos = zeros(npop, dim)
    PopFit = zeros(npop)

    newPopFit = zeros(npop, dim)
    his_best_fit = zeros(max_iter + 1)
    PopPos = initialization(npop, dim, ub, lb)
    for i = 1:npop
        PopFit[i] = objfun(PopPos[i, :])
    end

    indF = sortperm(PopFit, rev=true)
    PopPos = PopPos[indF, :]
    PopFit = PopFit[indF]
    his_best_fit[1] = BestF = PopFit[end]
    BestX = PopPos[end, :]

    Matr = [1, dim]

    for it = 1:max_iter
        r1 = rand()
        a = (1 - it / max_iter) * r1
        xrand = rand(dim) .* (ub - lb) .+ lb
        newPopPos = zeros(npop, dim)
        newPopPos[1, :] = (1 - a) * PopPos[end, :] .+ a * xrand  # equation (1)

        for i = 3:npop
            u = randn(dim)
            v = randn(dim)
            C = 1 / 2 * u ./ abs.(v)  # equation (4)

            r = rand()
            if r < 1 / 3
                newPopPos[i, :] = PopPos[i, :] .+ C .* (PopPos[i, :] .- newPopPos[1, :])  # equation (6)
            elseif r < 2 / 3
                r_idx = rand(2:i-1)
                newPopPos[i, :] = PopPos[i, :] .+ C .* (PopPos[i, :] .- PopPos[r_idx, :])  # equation (7)
            else
                r2 = rand()
                r_idx = rand(2:i-1)
                newPopPos[i, :] = PopPos[i, :] .+ C .* (r2 * (PopPos[i, :] .- newPopPos[1, :]) .+ (1 - r2) .* (PopPos[i, :] .- PopPos[r_idx, :]))  # equation (8)
            end
        end

        for i = 1:npop
            newPopPos[i, :] = max.(min.(newPopPos[i, :], ub), lb)
            newPopFit[i] = objfun(newPopPos[i, :])
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        indOne = argmin(PopFit)
        for i = 1:npop
            r3 = rand()
            Ind = rand(1:2)
            newPopPos[i, :] = PopPos[indOne, :] .+ 3 * randn(Matr[Ind]) .* ((r3 * rand(1:2) .- 1) .* PopPos[indOne, :] .- (2 * r3 - 1) .* PopPos[i, :])  # equation (9)
        end

        for i = 1:npop
            newPopPos[i, :] = max.(min.(newPopPos[i, :], ub), lb)
            newPopFit[i] = objfun(newPopPos[i, :])
            if newPopFit[i] < PopFit[i]
                PopPos[i, :] = newPopPos[i, :]
                PopFit[i] = newPopFit[i]
            end
        end

        indF = sortperm(PopFit, rev=true)
        PopPos = PopPos[indF, :]
        PopFit = PopFit[indF]

        if PopFit[end] < BestF
            BestF = PopFit[end]
            BestX = PopPos[end, :]
        end

        his_best_fit[it+1] = BestF
    end

    return OptimizationResult(
        BestX,
        BestF,
        his_best_fit)
end

function AEO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return AEO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end