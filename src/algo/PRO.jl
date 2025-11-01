"""
# References:

- Moosavi, Seyyed Hamid Samareh, and Vahid Khatibi Bardsiri.
  "Poor and rich optimization algorithm: A new human-based and multi populations algorithm."
  Engineering applications of artificial intelligence 86 (2019): 165-181.
"""
function PRO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return PRO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function PRO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    VarSize = (1, dim)

    npoor = div(npop, 2)
    nrich = div(npop, 2)
    pmut = 0.06

    # pop = [Dict("Position" => rand(dim) .* (ub - lb) .+ lb,
    #     "Cost" => Inf) for _ = 1:npop]
    pop = [Dict("Position" => initialization(1, dim, ub, lb),
        "Cost" => Inf) for _ = 1:npop]

    for i = 1:npop
        pop[i]["Cost"] = objfun(pop[i]["Position"])
    end
    pop = sort(pop, by=x -> x["Cost"])
    BestSol = pop[1]

    pop1 = pop[1:nrich]
    pop2 = pop[nrich+1:end]

    BestCost = zeros(max_iter)
    allpop = zeros(max_iter, npop)

    for it = 1:max_iter
        newpop1 = deepcopy(pop1)
        newpop2 = deepcopy(pop2)

        for i = 1:nrich
            newpop1[i]["Position"] += rand() .* (pop1[i]["Position"] - pop2[1]["Position"])

            if rand() < pmut
                newpop1[i]["Position"] += randn(dim)
            end

            newpop1[i]["Cost"] = objfun(newpop1[i]["Position"])
            newpop1[i]["Position"] = clamp.(newpop1[i]["Position"], lb, ub)
        end

        meanr = zeros(30)

        for d = 1:nrich
            meanr .+= pop1[d]["Position"]
        end

        meanr ./= nrich

        pattern = (pop1[1]["Position"] + meanr + pop1[end]["Position"]) / 3

        for j = 1:npoor
            newpop2[j]["Position"] += rand() .* (pattern - pop2[j]["Position"])

            if rand() < pmut
                newpop2[j]["Position"] += randn(dim)
            end

            newpop2[j]["Cost"] = objfun(newpop2[j]["Position"])
            newpop2[j]["Position"] = clamp.(newpop2[j]["Position"], lb, ub)
        end

        pop = vcat(pop1, pop2, newpop1, newpop2)

        pop = sort(pop, by=x -> x["Cost"])[1:npop]
        pop1 = pop[1:nrich]
        pop2 = pop[nrich+1:end]

        BestSol = pop1[1]
        BestCost[it] = BestSol["Cost"]
    end

    # return BestSol["Cost"], BestSol, BestCost[end]
    return OptimizationResult(
        BestSol,
        BestSol["Cost"],
        BestCost[end])
end

function PRO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return PRO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end