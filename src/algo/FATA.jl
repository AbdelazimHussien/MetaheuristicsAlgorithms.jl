"""
# References:

- Qi, Ailiang, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, and Huiling Chen.
  "FATA: an efficient optimization method based on geophysics."
  Neurocomputing 607 (2024): 128289.
"""
function FATA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return FATA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function FATA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    worstInte = 0
    bestInte = Inf
    arf = 0.2
    gBest = zeros(dim)
    bestPos = zeros(dim)
    cg_curve = Float64[]
    gBestScore = Inf
    Flight = initialization(npop, dim, ub, lb)
    fitness = fill(Inf, npop)
    it = 1
    FEs = 0
    lb = fill(lb, dim)
    ub = fill(ub, dim)
    MaxFEs = max_iter * dim

    while FEs < MaxFEs
        for i in axes(Flight, 1)
            Flight[i, :] .= max.(min.(Flight[i, :], ub), lb)
            FEs += 1
            fitness[i] = objfun(Flight[i, :])

            if gBestScore > fitness[i]
                gBestScore = fitness[i]
                gBest = Flight[i, :]
            end
        end

        Index = sortperm(fitness)
        Order = fitness[Index]
        worstFitness = Order[npop]
        bestFitness = Order[1]

        Integral = cumsum(Order)
        if Integral[npop] > worstInte
            worstInte = Integral[npop]
        end
        if Integral[npop] < bestInte
            bestInte = Integral[npop]
        end
        IP = (Integral[npop] - worstInte) / (bestInte - worstInte + eps())

        a = tan(-(FEs / MaxFEs) + 1)
        b = 1 / tan(-(FEs / MaxFEs) + 1)

        for i in axes(Flight, 1)
            Para1 = a * rand(dim) .- a * rand(dim)
            Para2 = b * rand(dim) .- b * rand(dim)
            p = (fitness[i] - worstFitness) / (gBestScore - worstFitness + eps())

            if rand() > IP
                Flight[i, :] = (ub .- lb) .* rand(dim) .+ lb
            else
                for j = 1:dim
                    num = rand(1:1:npop)
                    if rand() < p
                        Flight[i, j] = gBest[j] + Flight[i, j] * Para1[j]
                    else
                        Flight[i, j] = Flight[num, j] + Para2[j] * Flight[i, j]
                        Flight[i, j] = 0.5 * (arf + 1) * (lb[j] + ub[j]) - arf * Flight[i, j]
                    end
                end
            end
        end

        push!(cg_curve, gBestScore)
        it += 1
        bestPos = gBest
    end

    # return gBestScore, bestPos, cg_curve
    return OptimizationResult(
        bestPos,
        gBestScore,
        cg_curve)
end

function FATA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return FATA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end