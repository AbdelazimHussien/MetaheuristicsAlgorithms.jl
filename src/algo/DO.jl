# function levyF(npop, dim, beta)
#     sigma_u = (gamma(1 + beta) * sin(π * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
#     u = randn(npop, dim) * sigma_u
#     v = randn(npop, dim)
#     step = u ./ abs.(v) .^ (1 / beta)
#     return step
# end

"""
# References:

- Zhao, Shijie, Tianran Zhang, Shilin Ma, and Miao Chen.
  "Dandelion Optimizer: A nature-inspired metaheuristic algorithm for engineering applications."
  Engineering Applications of Artificial Intelligence 114 (2022): 105075.
"""
function DO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return DO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function DO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # dandelions = rand(npop, dim) .* (ub - lb) .+ lb
    dandelions = initialization(npop, dim, ub, lb)
    dandelionsFitness = zeros(npop)
    Convergence_curve = zeros(max_iter)

    for i = 1:npop
        dandelionsFitness[i] = objfun(dandelions[i, :])
    end

    sorted_indexes = sortperm(dandelionsFitness)
    Best_position = dandelions[sorted_indexes[1], :]
    Best_fitness = dandelionsFitness[sorted_indexes[1]]
    Convergence_curve[1] = Best_fitness

    t = 2

    while t <= max_iter
        beta = randn(npop, dim)
        alpha = rand() * ((1 / max_iter^2) * t^2 - 2 / max_iter * t + 1)
        a = 1 / (max_iter^2 - 2 * max_iter + 1)
        b = -2 * a
        c = 1 - a - b
        k = 1 - rand() * (c + a * t^2 + b * t)

        if randn() < 1.5
            dandelions_1 = copy(dandelions)
            for i = 1:npop
                lamb = abs.(randn(dim))
                theta = (2 * rand() - 1) * π
                row = 1 / exp(theta)
                vx = row * cos(theta)
                vy = row * sin(theta)
                NEW = rand(dim) .* (ub - lb) .+ lb
                dandelions_1[i, :] .= dandelions[i, :] .+ alpha .* vx .* vy .* logpdf.(Normal(0, 1), lamb) .* (NEW .- dandelions[i, :])
            end
        else
            dandelions_1 = copy(dandelions)
            for i = 1:npop
                dandelions_1[i, :] .= dandelions[i, :] .* k
            end
        end
        dandelions .= dandelions_1
        dandelions = clamp.(dandelions, lb, ub)

        dandelions_mean = sum(dandelions, dims=1) ./ npop
        dandelions_2 = copy(dandelions)
        for i = 1:npop
            for j = 1:dim
                dandelions_2[i, j] = dandelions[i, j] - beta[i, j] * alpha * (dandelions_mean[1, j] - beta[i, j] * alpha * dandelions[i, j])
            end
        end
        dandelions .= dandelions_2
        dandelions = clamp.(dandelions, lb, ub)

        # Step_length = levyF(npop, dim, 1.5)
        Step_length = levy(npop, dim, 1.5)
        Elite = repeat(Best_position', npop, 1)
        dandelions_3 = copy(dandelions)
        for i = 1:npop
            for j = 1:dim
                dandelions_3[i, j] = Elite[i, j] + Step_length[i, j] * alpha * (Elite[i, j] - dandelions[i, j] * (2 * t / max_iter))
            end
        end
        dandelions .= dandelions_3
        dandelions = clamp.(dandelions, lb, ub)

        for i = 1:npop
            dandelionsFitness[i] = objfun(dandelions[i, :])
        end

        sorted_indexes = sortperm(dandelionsFitness)
        dandelions = dandelions[sorted_indexes, :]
        SortfitbestN = dandelionsFitness[sorted_indexes]

        if SortfitbestN[1] < Best_fitness
            Best_position = dandelions[1, :]
            Best_fitness = SortfitbestN[1]
        end

        Convergence_curve[t] = Best_fitness
        t += 1
    end

    # return Best_fitness, Best_position, Convergence_curve
    return OptimizationResult(
        Best_position,
        Best_fitness,
        Convergence_curve)
end

function DO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return DO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end