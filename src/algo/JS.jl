"""
# References:

- Chou, Jui-Sheng, and Dinh-Nhat Truong.
  "A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean."
  Applied Mathematics and Computation 389 (2021): 125535.
"""
function JS(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return JS(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function JS(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    VarSize = dim
    if length(lb) == 1
        lb = lb .* ones(dim)
        ub = ub .* ones(dim)
    else
        lb = lb
        ub = ub
    end

    popi = initialization(npop, dim, ub, lb)
    it = 0

    BestSol = zeros(dim)

    fval = Inf

    popCost = zeros(npop)
    for i = 1:npop
        popCost[i] = objfun(popi[i, :])
    end

    fbestvl = zeros(max_iter)

    for it = 1:max_iter
        Meanvl = mean(popi, dims=1)[:]
        sorted_costs = sortperm(popCost)
        BestSol = popi[sorted_costs[1], :]
        BestCost = popCost[sorted_costs[1]]

        for i = 1:npop
            Ar = (1 - it * ((1) / max_iter)) * (2 * rand() - 1)

            if abs(Ar) >= 0.5
                newsol = popi[i, :] .+ rand(VarSize) .* (BestSol .- 3 * rand() * Meanvl)
                newsol = max.(min.(newsol, ub), lb)

                newsolCost = objfun(newsol)

                if newsolCost < popCost[i]
                    popi[i, :] = newsol
                    popCost[i] = newsolCost
                    if popCost[i] < BestCost
                        BestCost = popCost[i]
                        BestSol = popi[i, :]
                    end
                end
            else
                if rand() <= (1 - Ar)
                    j = rand(1:npop)
                    while j == i
                        j = rand(1:npop)
                    end
                    Step = popi[i, :] .- popi[j, :]
                    if popCost[j] < popCost[i]
                        Step = -Step
                    end
                    newsol = popi[i, :] .+ rand(VarSize) .* Step
                else
                    newsol = popi[i, :] .+ 0.1 * (ub .- lb) .* rand()
                end

                newsol = max.(min.(newsol, ub), lb)
                newsolCost = objfun(newsol)

                if newsolCost < popCost[i]
                    popi[i, :] = newsol
                    popCost[i] = newsolCost
                    if popCost[i] < BestCost
                        BestCost = popCost[i]
                        BestSol = popi[i, :]
                    end
                end
            end
        end

        fbestvl[it] = BestCost
    end

    u = BestSol
    fval = fbestvl[end]
    # return fval, u, fbestvl
    return OptimizationResult(
        u,
        fval,
        fbestvl)
end

function JS(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return JS(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end