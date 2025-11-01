mutable struct Mongoose
    Position::Vector{Float64}
    Cost::Float64
end

"""
# References:

- Agushaka, Jeffrey O., Absalom E. Ezugwu, and Laith Abualigah.
  "Dwarf mongoose optimization algorithm."
  Computer methods in applied mechanics and engineering 391 (2022): 114570.
"""
function DMOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return DMOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function DMOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    VarSize = dim

    nBabysitter = 3
    nAlphaGroup = npop - nBabysitter
    nScout = nAlphaGroup

    L = round(Int, 0.6 * dim * nBabysitter)
    peep = 2.0

    pop = [Mongoose(zeros(VarSize), Inf) for _ = 1:nAlphaGroup]

    BestSol = Mongoose(zeros(VarSize), Inf)
    tau = Inf
    sm = fill(Inf, nAlphaGroup)

    for i = 1:nAlphaGroup
        # pop[i].Position = rand(VarSize) .* (ub .- lb) .+ lb
        pop[i].Position = initialization(1, VarSize, ub, lb)
        pop[i].Cost = objfun(pop[i].Position)
        if pop[i].Cost <= BestSol.Cost
            BestSol = pop[i]
        end
    end

    C = zeros(Int, nAlphaGroup)
    CF = (1 - 1 / max_iter)^(2 * 1 / max_iter)

    BestCost = zeros(Float64, max_iter)

    for it = 1:max_iter
        F = zeros(Float64, nAlphaGroup)
        MeanCost = mean([pop[i].Cost for i = 1:nAlphaGroup])

        for i = 1:nAlphaGroup
            F[i] = exp(-pop[i].Cost / MeanCost)
        end

        P = F / sum(F)

        for m = 1:nAlphaGroup
            i = RouletteWheelSelection(P)

            K = setdiff(1:nAlphaGroup, i)
            k = K[rand(1:length(K))]

            phi = (peep / 2) * rand(VarSize) .* (2 .* rand(VarSize) .- 1)

            newpop = Mongoose(zeros(VarSize), Inf)
            newpop.Position = pop[i].Position .+ phi .* (pop[i].Position .- pop[k].Position)

            newpop.Cost = objfun(newpop.Position)

            if newpop.Cost <= pop[i].Cost
                pop[i] = newpop
            else
                C[i] += 1
            end
        end

        for i = 1:nScout
            K = setdiff(1:nAlphaGroup, i)
            k = K[rand(1:length(K))]

            phi = (peep / 2) * rand(VarSize) .* (2 .* rand(VarSize) .- 1)

            newpop = Mongoose(zeros(VarSize), Inf)
            newpop.Position = pop[i].Position .+ phi .* (pop[i].Position .- pop[k].Position)

            newpop.Cost = objfun(newpop.Position)

            sm[i] = (newpop.Cost - pop[i].Cost) / max(newpop.Cost, pop[i].Cost)

            if newpop.Cost <= pop[i].Cost
                pop[i] = newpop
            else
                C[i] += 1
            end
        end

        for i = 1:nBabysitter
            if C[i] >= L
                pop[i].Position = rand(VarSize) .* (ub .- lb) .+ lb
                pop[i].Cost = objfun(pop[i].Position)
                C[i] = 0
            end
        end

        for i = 1:nAlphaGroup
            if pop[i].Cost <= BestSol.Cost
                BestSol = pop[i]
            end
        end

        newtau = mean(sm)
        for i = 1:nScout
            M = (pop[i].Position .* sm[i]) ./ pop[i].Position
            if newtau > tau
                newpop = Mongoose(zeros(VarSize), Inf)
                newpop.Position = pop[i].Position .- CF * rand(VarSize) .* (pop[i].Position .- M)
            else
                newpop = Mongoose(zeros(VarSize), Inf)
                newpop.Position = pop[i].Position .+ CF * rand(VarSize) .* (pop[i].Position .- M)
            end
            tau = newtau
        end

        for i = 1:nAlphaGroup
            if pop[i].Cost <= BestSol.Cost
                BestSol = pop[i]
            end
        end

        BestCost[it] = BestSol.Cost
    end

    # return BestSol.Cost, BestSol.Position, BestCost
    return OptimizationResult(
        BestSol.Position,
        BestSol.Cost,
        BestCost)
end

function DMOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return DMOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end