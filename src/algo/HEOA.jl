"""
# References:

- Lian, Junbo, and Guohua Hui.
  "Human evolutionary optimization algorithm."
  Expert Systems with Applications 241 (2024): 122638.
"""
function HEOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return HEOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function HEOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    jump_factor = abs(lb[1] - ub[1]) / 1000

    A = 0.6
    LN = 0.4
    EN = 0.4
    FN = 0.1

    LNNumber = round(Int, npop * LN)
    ENNumber = round(Int, npop * EN)
    FNNumber = round(Int, npop * FN)

    if length(ub) == 1
        ub = fill(ub[1], dim)
        lb = fill(lb[1], dim)
    end

    X0 = initializationLogistic(npop, dim, ub, lb)
    X = X0

    fitness_new = zeros(npop)
    Best_pos = zeros(dim)
    Best_score = Inf

    fitness = [objfun(X[i, :]) for i = 1:npop]
    index = sortperm(fitness)
    fitness = fitness[index]
    BestF = fitness[1]
    WorstF = fitness[end]
    GBestF = fitness[1]
    AveF = mean(fitness)
    X = X0[index, :]
    curve = zeros(Float64, max_iter)
    avg_fitness_curve = zeros(Float64, max_iter)
    GBestX = X[1, :]
    X_new = copy(X)

    for i = 1:max_iter
        # if i % 100 == 0
        #     println("At iteration $i, the fitness is $(curve[i-1])")
        # end

        BestF = fitness[1]
        WorstF = fitness[end]
        avg_fitness_curve[i] = AveF
        R = rand()

        for j = 1:npop
            if i <= (1 / 4) * max_iter
                X_new[j, :] = GBestX * (1 - i / max_iter) +
                              (mean(X[j, :]) .- GBestX) * floor(rand() / jump_factor) * jump_factor +
                              0.2 * (1 - i / max_iter) .* (X[j, :] .- GBestX) .* levy(dim)
            else
                for j = 1:LNNumber
                    if R < A
                        X_new[j, :] = 0.2 * cos(pi / 2 * (1 - (i / max_iter))) * X[j, :] * exp((-i * randn()) / (rand() * max_iter))
                    else
                        X_new[j, :] = 0.2 * cos(pi / 2 * (1 - (i / max_iter))) * X[j, :] + randn() * ones(Float64, dim)
                    end
                end

                for j = (LNNumber+1):(LNNumber+ENNumber)
                    X_new[j, :] = randn(dim) .* exp.((X[end, :] - X[j, :]) / j^2)
                end

                for j = (LNNumber+ENNumber+1):(LNNumber+ENNumber+FNNumber)
                    X_new[j, :] = X[j, :] + 0.2 * cos(pi / 2 * (1 - (i / max_iter))) * rand(Float64, dim) .* (X[1, :] - X[j, :])
                end

                for j = (LNNumber+ENNumber+FNNumber+1):npop
                    X_new[j, :] = GBestX + (GBestX - X[j, :]) * randn()
                end
            end

            Flag4ub = X_new[j, :] .> ub
            Flag4lb = X_new[j, :] .< lb
            X_new[j, :] .= (X_new[j, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness_new[j] = objfun(vec(X_new[j, :]))
            if fitness_new[j] < GBestF
                GBestF = fitness_new[j]
                GBestX = X_new[j, :]
            end
        end

        X .= X_new
        fitness .= fitness_new
        curve[i] = GBestF
        Best_pos = GBestX
        Best_score = curve[end]
    end

    # return Best_score, Best_pos, curve
    return OptimizationResult(
        Best_pos,
        Best_score,
        curve)
end

function HEOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return HEOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function initializationLogistic(pop::Int, dim::Int, ub, lb)
    Boundary_no = length(ub)
    Positions = zeros(Float64, pop, dim)

    for i = 1:pop
        for j = 1:dim
            x0 = rand()
            a = 4.0
            x = a * x0 * (1 - x0)

            if Boundary_no == 1
                Positions[i, j] = (ub[1] - lb[1]) * x + lb[1]
                Positions[i, j] = clamp(Positions[i, j], lb[1], ub[1])
            else
                Positions[i, j] = (ub[j] - lb[j]) * x + lb[j]
                Positions[i, j] = clamp(Positions[i, j], lb[j], ub[j])
            end
            x0 = x
        end
    end

    return Positions
end