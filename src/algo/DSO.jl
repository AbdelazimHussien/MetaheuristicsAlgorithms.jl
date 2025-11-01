# struct DSOResult <: OptimizationResult
#     Best_Cost::Float64
#     Best_Position::Vector{Float64}
#     Best_iteration::Vector{Float64}
# end

"""
    DSO(npop, max_iter, lb, ub, dim, objfun)

The Deep Sleep Optimiser (DSO) is a human-inspired metaheuristic optimization algorithm that mimics the sleep patterns of humans to find optimal solutions in various optimization problems.

# Arguments:

- `npop`: Number of search agents (population size).
- `max_iter`: Number of iterations (generations).
- `lb`: Lower bounds of the search space.
- `ub`: Upper bounds of the search space.
- `objfun`: Objective function to be minimized.

# Returns:

- `::DSOResult`: A struct containing:
  - `Best_Cost`: The best cost found during the optimization.
  - `Best_Position`: The position corresponding to the best cost.
  - `Best_iteration`: A vector of best costs at each iteration.

# References:

- Oladejo, Sunday O., Stephen O. Ekwe, Lateef A. Akinyemi, and Seyedali A. Mirjalili,
  The deep sleep optimiser: A human-based metaheuristic approach., IEEE Access (2023).
"""
function DSO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return DSO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function DSO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    prob = objfun

    if length(ub) != length(lb)
        throw(ArgumentError("lb and ub must have the same length as dim."))
    end

    dim = length(lb)
    lb = lb
    ub = ub
    npop = npop
    max_iter = max_iter

    H0_minus = 0.17
    H0_plus = 0.85
    a = 0.1
    xs = 4.2
    xw = 18.2
    T = 24

    fit = zeros(npop)
    Best_iteration = zeros(max_iter + 1)

    C = sin((2 * π) / T)
    H_min = H0_minus + a * C
    H_max = H0_plus + a * C

    Pop = [lb .+ (ub .- lb) .* rand(dim) for _ = 1:npop]

    for q = 1:npop
        fit[q] = prob(Pop[q])
    end
    Best_iteration[1] = minimum(fit)

    for i = 1:max_iter
        for j = 1:npop
            Xini = Pop[j]
            _, ind = findmin(fit)
            Xbest = Pop[ind]

            mu = rand()
            mu = clamp(mu, H_min, H_max)
            H0 = Pop[j] .+ rand(dim) .* (Xbest .- mu .* Xini)

            if rand() > mu
                H = H0 .* 10^(-1 / xs)
            else
                H = mu .+ (H0 .- mu) .* 10^(-1 / xw)
            end

            H = clamp.(H, lb, ub)

            fnew = prob(H)
            if fnew < fit[j]
                Pop[j] = H
                fit[j] = fnew
            end
        end
        Best_iteration[i+1] = minimum(fit)
    end

    Best_Cost, idx = findmin(fit)
    Best_Position = Pop[idx]

    return OptimizationResult(
        Best_Position,
        Best_Cost,
        Best_iteration)
end

function DSO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return DSO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end