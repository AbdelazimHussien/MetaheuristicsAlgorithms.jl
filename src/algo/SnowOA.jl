"""
# References:

- Deng, Lingyun, and Sanyang Liu.
  "Snow ablation optimizer: A novel metaheuristic technique for numerical optimization and engineering design."
  Expert Systems with Applications 225 (2023): 120069.
"""
function SnowOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SnowOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SnowOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    if length(ub) == 1
        ub = ub * ones(dim)
        lb = lb * ones(dim)
    end

    X = initialization(npop, dim, ub, lb)

    best_pos = zeros(dim)
    best_score = Inf
    Objective_values = zeros(npop)

    convergence_curve = []

    N1 = floor(Int, npop * 0.5)
    Elite_pool = zeros(4, dim)

    for i = 1:npop
        Objective_values[i] = objfun(X[i, :])
        if i == 1
            best_pos = X[i, :]
            best_score = Objective_values[i]
        elseif Objective_values[i] < best_score
            best_pos = X[i, :]
            best_score = Objective_values[i]
        end
    end

    idx1 = sortperm(Objective_values)
    second_best = X[idx1[2], :]
    third_best = X[idx1[3], :]
    sum1 = sum(X[idx1[1:N1], :], dims=1)
    half_best_mean = sum1 / N1

    Elite_pool[1, :] = best_pos
    Elite_pool[2, :] = second_best
    Elite_pool[3, :] = third_best
    Elite_pool[4, :] = half_best_mean

    push!(convergence_curve, best_score)

    index = collect(1:npop)

    Na = npop ÷ 2
    Nb = npop ÷ 2

    l = 2
    while l <= max_iter
        RB = randn(npop, dim)
        T = exp(-l / max_iter)
        k = 1
        DDF = 0.35 * (1 + (5 / 7) * (exp(l / max_iter) - 1)^k / (exp(1) - 1)^k)
        M = DDF * T

        X_centroid = sum(X, dims=1) / npop

        index1 = randperm(npop)[1:Na]
        index2 = setdiff(index, index1)

        for i = 1:Na
            r1 = rand()
            k1 = rand(1:4)
            for j = 1:dim
                X[index1[i], j] = Elite_pool[k1, j] + RB[index1[i], j] * (r1 * (best_pos[j] - X[index1[i], j]) + (1 - r1) * (X_centroid[j] - X[index1[i], j]))
            end
        end

        if Na < npop
            Na += 1
            Nb -= 1
        end

        if Nb >= 1
            for i = 1:Nb
                r2 = 2 * rand() - 1
                for j = 1:dim
                    X[index2[i], j] = M * best_pos[j] + RB[index2[i], j] * (r2 * (best_pos[j] - X[index2[i], j]) + (1 - r2) * (X_centroid[j] - X[index2[i], j]))
                end
            end
        end

        for i = 1:npop
            X[i, :] = clamp.(X[i, :], lb, ub)
            Objective_values[i] = objfun(X[i, :])

            if Objective_values[i] < best_score
                best_pos = X[i, :]
                best_score = Objective_values[i]
            end
        end

        idx1 = sortperm(Objective_values)
        second_best = X[idx1[2], :]
        third_best = X[idx1[3], :]
        sum1 = sum(X[idx1[1:N1], :], dims=1)
        half_best_mean = sum1 / N1

        Elite_pool[1, :] = best_pos
        Elite_pool[2, :] = second_best
        Elite_pool[3, :] = third_best
        Elite_pool[4, :] = half_best_mean

        push!(convergence_curve, best_score)
        l += 1
    end

    # return best_score, best_pos, convergence_curve
    return OptimizationResult(
        best_pos,
        best_score,
        convergence_curve)
end

function SnowOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SnowOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end