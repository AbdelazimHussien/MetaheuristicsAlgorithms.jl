"""
# References:

- Lian, Junbo, Guohua Hui, Ling Ma, Ting Zhu, Xincan Wu, Ali Asghar Heidari, Yi Chen, and Huiling Chen.
  "Parrot optimizer: Algorithm and applications to medical problems."
  Computers in Biology and Medicine 172 (2024): 108064.
"""
function ParrotO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return ParrotO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function ParrotO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    if length(ub) == 1
        ub = fill(ub, dim)
        lb = fill(lb, dim)
    end

    X0 = initialization(npop, dim, ub, lb)
    X = X0

    fitness = zeros(npop)
    for i = 1:npop
        fitness[i] = objfun(X[i, :])
    end

    index = sortperm(fitness)
    fitness = fitness[index]
    GBestF = fitness[1]
    for i = 1:npop
        X[i, :] .= X0[index[i], :]
    end

    curve = zeros(max_iter)
    GBestX = X[1, :]
    X_new = copy(X)

    for i = 1:max_iter
        alpha = rand() / 5
        sita = rand() * π

        for i in axes(X, 1)
            St = rand(1:4)

            if St == 1
                X_new[j, :] .= (X[j, :] .- GBestX) .* levy(dim) .+ rand() * mean(X) * (1 - i / max_iter)^(2 * i / max_iter)

            elseif St == 2
                X_new[j, :] .= X[j, :] .+ GBestX .* levy(dim) .+ randn() * (1 - i / max_iter) * ones(dim)

            elseif St == 3
                H = rand()
                if H < 0.5
                    X_new[j, :] .= X[j, :] .+ alpha * (1 - i / max_iter) * (X[j, :] .- mean(X))
                else
                    X_new[j, :] .= X[j, :] .+ alpha * (1 - i / max_iter) * exp(-j / (rand() * max_iter))
                end

            else
                X_new[j, :] .= X[j, :] .+ rand() * cos((π * i) / (2 * max_iter)) * (GBestX .- X[j, :]) .- cos(sita) * (i / max_iter)^(2 / max_iter) * (X[j, :] .- GBestX)
            end

            for m = 1:npop
                for a = 1:dim
                    if X_new[m, a] > ub[a]
                        X_new[m, a] = ub[a]
                    end
                    if X_new[m, a] < lb[a]
                        X_new[m, a] = lb[a]
                    end
                end
            end

            if objfun(X_new[j, :]) < GBestF
                GBestF = objfun(X_new[j, :])
                GBestX .= X_new[j, :]
            end
        end

        fitness_new = zeros(npop)
        for s = 1:npop
            fitness_new[s] = objfun(X_new[s, :])
        end

        for s = 1:npop
            if fitness_new[s] < GBestF
                GBestF = fitness_new[s]
                GBestX .= X_new[s, :]
            end
        end

        X .= X_new
        fitness .= fitness_new

        index = sortperm(fitness)
        fitness = fitness[index]
        for s = 1:npop
            X0[s, :] .= X[index[s], :]
        end

        X .= X0
        curve[i] = GBestF
    end

    Best_pos = GBestX
    Best_score = curve[end]

    # return Best_score, Best_pos, curve
    return OptimizationResult(
        Best_pos,
        Best_score,
        curve)
end


function ParrotO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return ParrotO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end