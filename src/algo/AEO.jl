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

@views function AEO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # Matr = (1, dim)
    x = initialization2(npop, dim, ub, lb)
    x_new = zeros(dim, npop)
    x_rand = zeros(dim)
    y = zeros(npop)
    y_new = zeros(npop)
    his_best_fit = zeros(max_iter + 1)
    u = zeros(dim)
    v = zeros(dim)
    C = zeros(dim)

    for i = 1:npop
        y[i] = objfun(x[:, i])
    end

    indF = sortperm(y, rev=true)
    sort!(y, rev=true)
    Base.permutecols!!(x, indF)  # x = copy(x[:, indF])
    his_best_fit[1] = BestF = y[end]
    BestX = x[:, end]

    for it = 1:max_iter
        r1 = rand()
        a = (1 - it / max_iter) * r1
        initialization2!(x_rand, ub, lb)
        @. x_new[:, 1] = (1 - a) * x[:, end] + a * x_rand  # equation (1)

        for i = 3:npop
            randn!(u)
            randn!(v)
            @. C = 1 / 2 * u / abs(v)  # equation (4)

            r = rand()
            if r < 1 / 3
                @. x_new[:, i] = x[:, i] + C * (x[:, i] - x_new[:, 1])  # equation (6)
            elseif r < 2 / 3
                r_idx = rand(2:i-1)
                @. x_new[:, i] = x[:, i] + C * (x[:, i] - x[:, r_idx])  # equation (7)
            else
                r2 = rand()
                r_idx = rand(2:i-1)
                @. x_new[:, i] = x[:, i] + C * (r2 * (x[:, i] - x_new[:, 1]) + (1 - r2) * (x[:, i] - x[:, r_idx]))  # equation (8)
            end
        end

        for i = 1:npop
            x_new[:, i] .= clamp.(x_new[:, i], lb, ub)
            y_new[i] = objfun(x_new[:, i])
            if y_new[i] < y[i]
                x[:, i] .= x_new[:, i]
                y[i] = y_new[i]
            end
        end

        indOne = argmin(y)
        for i = 1:npop
            r3 = rand()
            rand(1:2) == 1 ? fill!(u, randn()) : randn!(u)
            @. x_new[:, i] = x[:, indOne] + 3 * u * ((r3 * $rand(1:2) - 1) * x[:, indOne] - (2 * r3 - 1) * x[:, i])  # equation (9)
        end

        for i = 1:npop
            x_new[:, i] .= clamp.(x_new[:, i], lb, ub)
            y_new[i] = objfun(x_new[:, i])
            if y_new[i] < y[i]
                x[:, i] .= x_new[:, i]
                y[i] = y_new[i]
            end
        end

        sortperm!(indF, y, rev=true)
        sort!(y, rev=true)
        Base.permutecols!!(x, indF)

        if y[end] < BestF
            BestF = y[end]
            BestX .= x[:, end]
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