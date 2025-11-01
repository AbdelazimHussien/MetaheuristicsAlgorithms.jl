"""
# References:

- Ghasemi, Mojtaba, Mohsen Zare, Amir Zahedi, Mohammad-Amin Akbari, Seyedali Mirjalili, and Laith Abualigah.
  "Geyser inspired algorithm: a new geological-inspired meta-heuristic for real-parameter and constrained engineering optimization."
  Journal of Bionic Engineering 21, no. 1 (2024): 374-408.
"""
function GEA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GEA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GEA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    VarSize = (1, dim)
    Nc = Int(floor(npop / 3))
    FEs = npop * max_iter

    # Positions = rand(npop, dim) * (ub - lb) .+ lb
    Positions = initialization(npop, dim, ub, lb)
    Costs = zeros(Float64, npop)
    for i = 1:npop
        Costs[i] = objfun(Positions[i, :])
    end
    BestCost = minimum(Costs)
    BestPosition = Positions[argmin(Costs)]
    BestCosts = zeros(FEs)

    it = npop

    while it < FEs
        for i = 1:npop
            D1 = [sum(Positions[i, :] .* Positions[j, :]) / sqrt(sum(Positions[i, :] .^ 2) * sum(Positions[j, :] .^ 2)) for j = 1:npop]
            D1[i] = Inf
            S1 = minimum(D1)
            j1 = argmin(D1)

            CS = sort(Costs)
            dif = CS[end] - CS[1]
            G = (Costs[i] - CS[1]) / dif
            G = dif == 0 ? 0 : G

            P_i = sqrt(((G^(2 / it)) - (G^((it + 1) / it))) * (it / (it - 1)))

            ImpFitness = sum(Costs[1:Nc])
            p = [Costs[ii] / ImpFitness for ii = 1:Nc]
            ImpFitness = ImpFitness == 0 ? 1e-320 : ImpFitness

            i1 = sum(p) == 0 ? 1 : RouletteWheelSelectionFun(p)

            flag = 0
            newsol_Position = Positions[j1, :] .+ rand(dim) .* (Positions[i1, :] - Positions[j1, :]) .+ rand(dim) .* (Positions[i1, :] - Positions[i, :])
            newsol_Position = max.(newsol_Position, lb)
            newsol_Position = min.(newsol_Position, ub)
            newsol_Cost = objfun(newsol_Position)
            it += 1

            if it > FEs
                break
            end

            if newsol_Cost <= Costs[i]
                Positions[i, :] = newsol_Position
                Costs[i] = newsol_Cost
            end
            if newsol_Cost <= BestCost
                BestCost = newsol_Cost
                BestPosition = newsol_Position
            end
            BestCosts[it] = BestCost

            if flag == 0
                i2 = RouletteWheelSelectionFun(1 .- p)
                newsol_Position = Positions[i2, :] .+ rand() * (P_i - rand()) .* rand(dim) * (ub - lb) .+ lb
                newsol_Position = max.(newsol_Position, lb)
                newsol_Position = min.(newsol_Position, ub)
                newsol_Cost = objfun(newsol_Position)
                it += 1

                if newsol_Cost <= Costs[i]
                    Positions[i, :] = newsol_Position
                    Costs[i] = newsol_Cost
                end
                if newsol_Cost <= BestCost
                    BestCost = newsol_Cost
                    BestPosition = newsol_Position
                end
                BestCosts[it] = BestCost
            end
        end

        sorted_indices = sortperm(Costs)
        Positions = Positions[sorted_indices, :]
        Costs = Costs[sorted_indices]
    end
    # return BestCost, BestPosition, BestCosts
    return OptimizationResult(
        BestPosition,
        BestCost,
        BestCosts)
end

function GEA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GEA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function RouletteWheelSelectionFun(p)
    c = cumsum(p)
    r = rand()
    i = findfirst(x -> x >= r, c)
    return i
end