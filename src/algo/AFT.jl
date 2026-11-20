"""
    AFT(noThieves, max_iter, lb, ub, objfun)

    Ali Baba and the Forty Thieves (AFT) meta-heuristic optimization algorithm implementation in Julia.

# Arguments:

- `noThieves`: Number of thieves in the algorithm.
- `max_iter`: Maximum number of iterations.
- `lb`: Lower bounds for the search space.
- `ub`: Upper bounds for the search space.
- `objfun`: Function to evaluate the fitness of solutions.

# Returns:

- `AFTResult`: A struct containing:
  - `fitness`: The best fitness value found.
  - `gbest`: The position corresponding to the best fitness.
  - `ccurve`: A vector of best fitness values at each iteration.

# References:

- Braik, Malik, Mohammad Hashem Ryalat, and Hussein Al-Zoubi. "A novel meta-heuristic algorithm for solving numerical optimization problems: Ali Baba and the forty thieves." Neural Computing and Applications 34, no. 1 (2022): 409-455.
"""
function AFT(objfun, lb::Real, ub::Real, noThieves::Integer, max_iter::Integer, dim::Integer)
    return AFT(objfun, fill(lb, dim), fill(ub, dim), noThieves, max_iter)
end

function AFT(objfun, lb::Vector{Float64}, ub::Vector{Float64}, noThieves::Integer, max_iter::Integer)::OptimizationResult
    dim = length(lb)

    ccurve = zeros(max_iter)

    xth = initialization(noThieves, dim, ub, lb)  # Position of the thieves in the space

    # Evaluate fitness of the initial population
    fit = zeros(noThieves)
    for i = 1:noThieves
        fit[i] = objfun(xth[i, :])
    end

    # Initialization of AFT parameters
    fitness = copy(fit)
    sorted_indexes = sortperm(fit)
    sorted_thieves_fitness = fit[sorted_indexes]

    Sorted_thieves = xth[sorted_indexes, :]
    gbest = Sorted_thieves[1, :]
    fit0 = sorted_thieves_fitness[1]

    best = copy(xth)  # Best positions of thieves
    xab = copy(xth)   # Position of Ali Baba

    # Start the AFT main loop
    for ite = 1:max_iter
        Pp = 0.1 * log(2.75 * (ite / max_iter)^0.1)
        Td = 2 * exp(-2 * (ite / max_iter)^2)

        a = Int.(ceil.((noThieves - 1) .* rand(noThieves))')

        for i = 1:noThieves
            if rand() >= 0.5
                if rand() > Pp
                    xth[i, :] = gbest .+ (Td .* (best[i, :] .- xab[i, :]) .* rand() .+ Td .* (xab[i, :] .- best[a[i], :]) .* rand()) .* sign(rand() - 0.50)
                else
                    xth[i, :] .= Td * ((ub .- lb) .* rand(dim) .+ lb)  # Case 3
                end
            else
                for j = 1:dim
                    xth[i, j] = gbest[j] .- (Td .* (best[i, j] .- xab[i, j]) .* rand() .+ Td .* (xab[i, j] .- best[a[i], j]) .* rand()) .* sign(rand() - 0.50)
                end
            end
        end

        for i = 1:noThieves
            fit[i] = objfun(xth[i, :])

            if all(xth[i, :] .>= lb) && all(xth[i, :] .<= ub)
                xab[i, :] = xth[i, :]

                if fit[i] < fitness[i]
                    best[i, :] = xth[i, :]
                    fitness[i] = fit[i]
                end

                if fitness[i] < fit0
                    fit0 = fitness[i]
                    gbest = best[i, :]
                end
            end
        end

        ccurve[ite] = fit0
    end

    bestThieves = findall(x -> x == minimum(fitness), fitness)
    gbestSol = best[bestThieves[1], :]
    fitness = objfun(gbestSol)


    # return OptimizationResult( 
    #     gbestSol, 
    #     fitness,
    #     ccurve)
    println("gbestSol ", gbestSol)
    println("fitness ", fitness)
    println("ccurve ", size(ccurve))

    return OptimizationResult(
        gbestSol,
        fit0,
        ccurve)
end

function AFT(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return AFT(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end