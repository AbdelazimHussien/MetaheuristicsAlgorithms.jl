"""
# References:

- Fu, Shengwei, et al.
  "Red-billed blue magpie optimizer: a novel metaheuristic algorithm for 2D/3D UAV path planning and engineering design problems."
  Artificial Intelligence Review 57.6 (2024): 134.
"""
function RBMO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return RBMO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end
function RBMO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    D = length(lb)
    Xfood = zeros(D)
    BestValue = Inf
    Conv = zeros(max_iter)
    fitness = fill(Inf, npop)
    FES = 0
    Epsilon = 0.5

    X = initialization(npop, D, ub, lb)
    X_old = copy(X)
    fitness_old = zeros(npop)

    for i = 1:npop
        fitness_old[i] = objfun(X_old[i, :])
    end

    t = 1

    while t ≤ max_iter
        for i = 1:npop
            p = rand(2:5)
            selected_index_p = randperm(npop)[1:p]
            Xp = X[selected_index_p, :]
            Xpmean = mean(Xp, dims=1)

            q = rand(10:npop)
            selected_index_q = randperm(npop)[1:q]
            Xq = X[selected_index_q, :]
            Xpmean = vec(mean(Xp, dims=1))

            A = randperm(npop)
            R1 = A[1]

            if rand() < Epsilon
                X[i, :] = X[i, :] .+ (Xpmean - X[R1, :]) .* rand(D)
            else
                X[i, :] = X[i, :] .+ (Xpmean - X[R1, :]) .* rand(D)
            end
        end

        X = max.(lb, min.(X, ub))

        for i = 1:npop
            fitness[i] = objfun(X[i, :])
            if fitness[i] < BestValue
                BestValue = fitness[i]
                Xfood = copy(X[i, :])
            end
            FES += 1
        end

        fitness, X, fitness_old, X_old = Food_storage(fitness, X, fitness_old, X_old)

        CF = (1 - t / max_iter)^(2 * t / max_iter)

        for i = 1:npop
            p = rand(2:5)
            selected_index_p = randperm(npop)[1:p]
            Xp = X[selected_index_p, :]
            Xpmean = vec(mean(Xp, dims=1))

            q = rand(10:npop)
            selected_index_q = randperm(npop)[1:q]
            Xq = X[selected_index_q, :]
            Xqmean = vec(mean(Xq, dims=1))

            if rand() < Epsilon
                X[i, :] .= Xfood .+ CF .* (Xpmean - X[i, :]) .* randn(D)
            else
                X[i, :] .= Xfood .+ CF .* (Xqmean - X[i, :]) .* randn(D)
            end
        end

        X = max.(lb, min.(X, ub))

        for i = 1:npop
            fitness[i] = objfun(X[i, :])
            if fitness[i] < BestValue
                BestValue = fitness[i]
                Xfood = copy(X[i, :])
            end
            FES += 1
        end

        fitness, X, fitness_old, X_old = Food_storage(fitness, X, fitness_old, X_old)

        Conv[t] = BestValue
        t += 1
    end

    # return BestValue, Xfood, Conv, FES
    return OptimizationResult(
        Xfood,
        BestValue,
        Conv)
end

function RBMO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return RBMO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function Food_storage(fit, X, fit_old, X_old)
    Inx = fit_old .< fit
    Indx = repeat(Inx, 1, size(X, 2))

    X = Indx .* X_old .+ .!Indx .* X
    fit = Inx .* fit_old .+ .!Inx .* fit

    fit_old = copy(fit)
    X_old = copy(X)

    return fit, X, fit_old, X_old
end

