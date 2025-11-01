"""
# References:

- Trojovská, Eva, Mohammad Dehghani, and Pavel Trojovský.
  "Zebra optimization algorithm: A new bio-inspired optimization algorithm for solving optimization algorithm."
  Ieee Access 10 (2022): 49445-49473.
"""
function ZOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return ZOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function ZOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    if length(lb) == 1 && length(ub) == 1
        lb = fill(lb, dim)
        ub = fill(ub, dim)
    end

    X = zeros(npop, dim)
    X = initialization(npop, dim, ub, lb)
    # for i = 1:dim
    #     X[:, i] = lb[i] .+ rand(npop) .* (ub[i] - lb[i])
    # end

    fit = zeros(npop)
    for i = 1:npop
        L = X[i, :]
        fit[i] = objfun(L)
    end

    best_so_far = zeros(max_iter)
    average = zeros(max_iter)
    fbest = Inf
    PZ = zeros(dim)

    for t = 1:max_iter
        best, location = findmin(fit)
        if t == 1 || best < fbest
            fbest = best
            PZ = X[location, :]
        end

        for i = 1:npop
            I = round(Int, 1 + rand())
            X_newP1 = X[i, :] + rand(dim) .* (PZ - I * X[i, :])
            X_newP1 = clamp.(X_newP1, lb, ub)

            f_newP1 = objfun(X_newP1)
            if f_newP1 <= fit[i]
                X[i, :] = X_newP1
                fit[i] = f_newP1
            end
        end

        Ps = rand()
        k = rand(1:npop)
        AZ = X[k, :]

        for i = 1:npop
            if Ps < 0.5
                R = 0.1
                X_newP2 = X[i, :] + R * (2 * rand(dim) .- 1) * (1 - t / max_iter) .* X[i, :]  # Eq(5) S1
                X_newP2 = clamp.(X_newP2, lb, ub)
            else
                I = round(Int, 1 + rand())
                X_newP2 = X[i, :] + rand(dim) .* (AZ - I * X[i, :])
                X_newP2 = clamp.(X_newP2, lb, ub)
            end

            f_newP2 = objfun(X_newP2)
            if f_newP2 <= fit[i]
                X[i, :] = X_newP2
                fit[i] = f_newP2
            end
        end

        best_so_far[t] = fbest
        average[t] = mean(fit)
    end

    best_score = fbest
    best_pos = PZ
    ZOA_curve = best_so_far

    # return best_score, best_pos, ZOA_curve
    return OptimizationResult(
        best_pos,
        best_score,
        ZOA_curve)
end

function ZOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)
    return ZOA(npop, max_iter, problem.lb, problem.ub, problem.dim, problem.objfun)
end