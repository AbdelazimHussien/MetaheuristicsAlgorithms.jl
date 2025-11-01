"""
# References:

- Ghasemi, Mojtaba, Mohsen Zare, Amir Zahedi, Pavel Trojovský, Laith Abualigah, and Eva Trojovská.
  "Optimization based on performance of lungs in body: Lungs performance-based optimization (LPO)."
  Computer Methods in Applied Mechanics and Engineering 419 (2024): 116582.
"""
function LPO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return LPO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function LPO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Positions = initialization(npop, dim, lb, ub)
    Fitness = fill(Inf, npop)
    BestPosition = zeros(dim)
    BestCost = Inf
    Convergence = zeros(Float64, max_iter)
    Delta = rand(npop) .* 2π
    sigma1 = [rand(1, dim) for _ = 1:npop]

    # Evaluate initial population
    for i = 1:npop
        Fitness[i] = objfun(Positions[i, :])  # Call cost function
        if Fitness[i] <= BestCost
            BestPosition = Positions[i, :]
            BestCost = Fitness[i]
        end
    end

    it = 0
    Pos1 = zeros(dim)
    while it <= max_iter
        BestCost = minimum(Fitness)
        WorstCost = maximum(Fitness)

        for i = 1:npop
            for jj = 1:5
                R = Fitness[i]
                C = (R / 2) * sin(Delta[i])

                newsol = deepcopy(Positions[i, :])
                newsol2 = deepcopy(Positions[i, :])

                if jj == 1
                    newsol .= Positions[i, :] .+ ((R^2 + (1 / (2π * dim * R * C)^2))^-0.5) * sin(2π * dim * it) * sin((2π * dim * it) + Delta[i]) .* Positions[i, :]
                else
                    newsol .= Positions[i, :] .+ ((R^2 + (1 / (2π * dim * R * C)^2))^-0.5) * sin(2π * dim * it) * sin((2π * dim * it) + Delta[i]) .* Pos1
                end

                perm = randperm(npop)
                a1, a2, a3 = perm[1], perm[2], perm[3]

                aa1 = (Fitness[a2] - Fitness[a3]) / abs(Fitness[a3] - Fitness[a2])
                aa1 = iszero(Fitness[a2] - Fitness[a3]) ? 1.0 : aa1

                aa2 = (Fitness[a1] - Fitness[i]) / abs(Fitness[a1] - Fitness[i])
                aa2 = iszero(Fitness[a1] - Fitness[i]) ? 1.0 : aa2

                newsol = newsol .+ aa2 * sigma1[i] .* (newsol .- Positions[a1, :]) + aa1 * sigma1[i] .* (Positions[a3, :] .- Positions[a2, :])
                newsol2 = Positions[a1, :] .+ sigma1[i] .* (Positions[a3, :] .- Positions[a2, :])

                for j = 1:dim
                    Pos1[j] = rand() / jj > rand() ? newsol2[j] : newsol[j]
                end

                Pos1 = clamp.(Pos1, lb, ub)
                newsol = Pos1
                Delta[i] = atan(1 / (2π * dim * R * C))

                newCost = objfun(newsol)
                if newCost < Fitness[i]
                    Positions[i, :] = newsol
                    Fitness[i] = newCost
                    if newCost <= BestCost
                        BestPosition = newsol
                        BestCost = newCost
                    end
                end
                sigma1[i] = rand(1, dim)
            end
        end

        it += 1
    end

    Result = Convergence

    # return BestCost, BestPosition, Convergence
    return OptimizationResult(
        BestPosition,
        BestCost,
        Convergence)
end

function LPO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return LPO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end