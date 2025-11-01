"""
# References:

- Mirjalili, Seyedali, Amir H. Gandomi, Seyedeh Zahra Mirjalili, Shahrzad Saremi, Hossam Faris, and Seyed Mohammad Mirjalili.
  "Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems."
  Advances in engineering software 114 (2017): 163-191.
"""
function SSA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SSA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SSA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)#(npop, max_iter, lb, ub, dim, objfun)#(npop, max_iter, objfun, dim, lb, ub)
    if size(ub, 1) == 1
        ub = ones(dim) * ub
        lb = ones(dim) * lb
    end

    Convergence_curve = zeros(max_iter)

    SalpPositions = initialization(npop, dim, ub, lb)

    food_position = zeros(dim)
    food_fitness = Inf

    SalpFitness = zeros(npop)
    for i in axes(SalpPositions, 1)
        SalpFitness[i] = objfun(SalpPositions[i, :])
    end

    sorted_indexes = sortperm(SalpFitness)
    Sorted_salps = SalpPositions[sorted_indexes, :]

    food_position .= Sorted_salps[1, :]
    food_fitness = SalpFitness[sorted_indexes[1]]

    l = 2
    while l <= max_iter
        c1 = 2 * exp(-((4 * l / max_iter)^2))
        for i = 1:size(SalpPositions, 1)
            SalpPositions = SalpPositions'

            if i <= npop / 2
                for j = 1:dim
                    c2 = rand()
                    c3 = rand()
                    if c3 < 0.5
                        SalpPositions[i, j] = food_position[j] + c1 * ((ub[j] - lb[j]) * c2 + lb[j])
                    else
                        SalpPositions[i, j] = food_position[j] - c1 * ((ub[j] - lb[j]) * c2 + lb[j])
                    end
                end
            elseif i > npop / 2 && i < npop + 1
                point1 = SalpPositions[:, i-1]
                point2 = SalpPositions[:, i]
                SalpPositions[:, i] = (point2 .+ point1) / 2
            end

            SalpPositions = SalpPositions'
        end

        for i in axes(SalpPositions, 1)
            SalpPositions[i, :] = max.(min.(SalpPositions[i, :], ub), lb)

            SalpFitness[i] = objfun(SalpPositions[i, :])

            if SalpFitness[i] < food_fitness
                food_position .= SalpPositions[i, :]
                food_fitness = SalpFitness[i]
            end
        end
        Convergence_curve[l] = food_fitness
        l += 1
    end

    # return food_fitness, food_position, Convergence_curve
    return OptimizationResult(
        food_position,
        food_fitness,
        Convergence_curve)
end

function SSA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SSA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end