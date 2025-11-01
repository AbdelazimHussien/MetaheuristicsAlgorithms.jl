"""
# References:

- Moosavi, Seyyed Hamid Samareh, and Vahid Khatibi Bardsiri.
  "Satin bowerbird optimizer: A new optimization algorithm to optimize ANFIS for software development effort estimation."
  Engineering Applications of Artificial Intelligence 60 (2017): 1-15.
"""
function SBO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SBO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SBO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    alpha = 0.94

    pMutation = 0.05

    Z = 0.02
    sigma = Z * (max(ub) - min(lb))

    pop = [Dict("Position" => zeros(dim), "Cost" => Inf) for _ = 1:npop]

    for i = 1:npop
        # pop[i]["Position"] = rand(dim) .* (ub - lb) .+ lb
        pop[i]["Position"] = initialization(1, dim, lb, ub)
        pop[i]["Cost"] = objfun(pop[i]["Position"])
    end

    pop = sort(pop, by=x -> x["Cost"])

    BestSol = pop[1]
    elite = BestSol["Position"]

    BestCost = zeros(max_iter)

    for it = 1:max_iter
        newpop = deepcopy(pop)

        F = [
            if p["Cost"] >= 0
                1 / (1 + p["Cost"])
            else
                1 + abs(p["Cost"])
            end for p in pop
        ]

        P = F / sum(F)

        for i = 1:npop
            for k = 1:dim
                j = RouletteWheelSelection(P)

                λ = alpha / (1 + P[j])

                newpop[i]["Position"][k] = pop[i]["Position"][k] +
                                           λ * ((pop[j]["Position"][k] + elite[k]) / 2 - pop[i]["Position"][k])

                if rand() <= pMutation
                    newpop[i]["Position"][k] += sigma * randn()
                end
            end

            newpop[i]["Cost"] = objfun(newpop[i]["Position"])
        end

        combined_pop = vcat(pop, newpop)
        combined_pop = sort!(combined_pop, by=x -> x["Cost"])
        pop = combined_pop[1:npop]

        BestSol = pop[1]
        elite = BestSol["Position"]

        BestCost[it] = BestSol["Cost"]
    end
    # return BestSol["Cost"], BestSol["Position"], BestCost
    return OptimizationResult(
        BestSol["Position"],
        BestSol["Cost"],
        BestCost)
end

function SBO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SBO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end
