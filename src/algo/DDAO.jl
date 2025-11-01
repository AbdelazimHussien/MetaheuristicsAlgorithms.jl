"""
# References:

- Ghafil, H. N., & Jármai, K. (2020).
  Dynamic differential annealed optimization: New metaheuristic optimization algorithm for engineering applications.
  Applied Soft Computing, 93, 106392.
"""
function DDAO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return DDAO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function DDAO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    VarLength = dim

    MaxSubIt = 1000
    T0 = 2000.0
    alpha = 0.995

    empty_template = Dict("Phase" => [], "Cost" => Inf)
    pop = [deepcopy(empty_template) for _ = 1:npop]

    BestSol = Dict("Phase" => [], "Cost" => Inf)

    for i = 1:npop
        # pop[i]["Phase"] = lb .+ rand(VarLength) .* (ub - lb)
        pop[i]["Phase"] = initialization(1, dim, ub, lb)

        pop[i]["Cost"] = objfun(pop[i]["Phase"])

        if pop[i]["Cost"] <= BestSol["Cost"]
            BestSol = deepcopy(pop[i])
        end
    end

    BestCost = zeros(max_iter)

    T = T0

    for t = 1:max_iter
        newpop = [deepcopy(empty_template) for _ = 1:MaxSubIt]

        for subit = 1:MaxSubIt
            newpop[subit]["Phase"] = lb .+ rand(VarLength) .* (ub - lb)
            newpop[subit]["Phase"] = clamp.(newpop[subit]["Phase"], lb, ub)
            newpop[subit]["Cost"] = objfun(newpop[subit]["Phase"])
        end

        SortOrder = sortperm([newpop[subit]["Cost"] for subit = 1:MaxSubIt])
        newpop = newpop[SortOrder]
        bnew = newpop[1]
        kk = rand(1:npop)
        bb = rand(1:npop)

        Mnew = deepcopy(empty_template)
        if isodd(t)
            Mnew["Phase"] = (pop[kk]["Phase"] - pop[bb]["Phase"]) + bnew["Phase"]
        else
            Mnew["Phase"] = (pop[kk]["Phase"] - pop[bb]["Phase"]) + bnew["Phase"] * rand()
        end

        Mnew["Phase"] = clamp.(Mnew["Phase"], lb, ub)

        Mnew["Cost"] = objfun(Mnew["Phase"])

        for i = 1:npop
            if Mnew["Cost"] <= pop[i]["Cost"]
                pop[i] = deepcopy(Mnew)
            else
                DELTA = Mnew["Cost"] - pop[i]["Cost"]
                P = exp(-DELTA / T)
                if rand() <= P
                    pop[end] = deepcopy(Mnew)
                end
            end

            if pop[i]["Cost"] <= BestSol["Cost"]
                BestSol = deepcopy(pop[i])
            end
        end

        T *= alpha
    end

    y = BestSol["Cost"]

    # return BestSol["Cost"], BestSol["Phase"], BestCost
    return OptimizationResult(
        BestSol["Phase"],
        BestSol["Cost"],
        BestCost)
end

function DDAO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return DDAO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end
