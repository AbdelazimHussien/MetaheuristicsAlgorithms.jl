mutable struct individual
    Position::Vector{Float64}
    Cost::Float64
end

"""
# References:

- Rao, R. Venkata, Vimal J. Savsani, and Dipakkumar P. Vakharia.
  "Teaching–learning-based optimization: a novel method for constrained mechanical design optimization problems."
  Computer-aided design 43, no. 3 (2011): 303-315.
"""
function TLBO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return TLBO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function TLBO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    VarSize = (dim)

    pop = [individual(rand(VarSize) * (ub - lb) .+ lb,
        objfun(vec(rand(VarSize) .* (ub - lb) .+ lb))) for _ = 1:npop]

    best_sol = individual([], Inf)

    for i = 1:npop
        pop[i].Cost = objfun(vec(pop[i].Position))
        if pop[i].Cost < best_sol.cost
            best_sol = pop[i]
        end
    end

    best_costs = zeros(max_iter)

    for it = 1:max_iter
        Mean = sum(p.Position for p in pop) / npop

        Teacher = pop[1]
        for i = 2:npop
            if pop[i].Cost < Teacher.Cost
                Teacher = pop[i]
            end
        end

        for i = 1:npop
            newsol = individual(zeros(VarSize), 0.0)

            TF = rand(1:2)

            newsol.Position = pop[i].Position .+ rand(VarSize) .* (Teacher.Position .- TF * Mean)

            newsol.Position = clamp.(newsol.Position, lb, ub)

            newsol.Cost = objfun(newsol.Position)

            if newsol.Cost < pop[i].Cost
                pop[i] = newsol
                if pop[i].Cost < best_sol.cost
                    best_sol = pop[i]
                end
            end
        end

        for i = 1:npop
            A = collect(1:npop)
            deleteat!(A, i)
            j = A[rand(1:length(A))]

            Step = pop[i].Position .- pop[j].Position
            if pop[j].Cost < pop[i].Cost
                Step .= -Step
            end

            newsol = individual(zeros(VarSize), 0.0)

            newsol.Position = pop[i].Position .+ rand(VarSize) .* Step

            newsol.Position = clamp.(newsol.Position, lb, ub)

            newsol.Cost = objfun(newsol.Position)

            if newsol.Cost < pop[i].Cost
                pop[i] = newsol
                if pop[i].Cost < best_sol.cost
                    best_sol = pop[i]
                end
            end
        end

        best_costs[it] = best_sol.cost
    end

    # return best_sol.cost, best_sol, best_costs
    return OptimizationResult(
        best_sol,
        best_sol.cost,
        best_costs)
end

function TLBO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return TLBO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end