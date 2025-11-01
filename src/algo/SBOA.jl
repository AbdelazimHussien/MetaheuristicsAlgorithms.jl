"""
# References:

- Fu, Youfa, Dan Liu, Jiadui Chen, and Ling He.
  "Secretary bird optimization algorithm: a new metaheuristic for solving global optimization problems."
  Artificial Intelligence Review 57, no. 5 (2024): 1-102.
"""
function SBOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SBOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end
function SBOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    lb = fill(lb, dim)
    ub = fill(ub, dim)

    X = initialization(npop, dim, lb, ub)
    fit = zeros(npop)
    Bast_P = zeros(dim)
    for i = 1:npop
        fit[i] = objfun(vec(X[i, :]))
    end
    SBOA_curve = zeros(max_iter)
    fbest = Inf

    for t = 1:max_iter
        CF = (1 - t / max_iter)^(2 * t / max_iter)

        best, location = findmin(fit)
        if t == 1
            Bast_P = X[location, :]
            fbest = best
        elseif best < fbest
            fbest = best
            Bast_P = X[location, :]
        end

        for i = 1:npop
            if t < max_iter / 3
                X_random_1 = X[rand(1:npop), :]
                X_random_2 = X[rand(1:npop), :]
                R1 = rand(dim)
                X1 = X[i, :] .+ (X_random_1 .- X_random_2) .* R1
                X1 = clamp.(X1, lb, ub)
            elseif t > max_iter / 3 && t < 2 * max_iter / 3
                RB = randn(dim)
                X1 = Bast_P .+ exp((t / max_iter)^4) .* (RB .- 0.5) .* (Bast_P .- X[i, :])
                X1 = clamp.(X1, lb, ub)
            else
                RL = 0.5 * levy(dim)
                X1 = Bast_P .+ CF .* X[i, :] .* RL
                X1 = clamp.(X1, lb, ub)
            end

            f_newP1 = objfun(X1)
            if f_newP1 <= fit[i]
                X[i, :] = X1
                fit[i] = f_newP1
            end
        end

        r = rand()
        Xrandom = X[rand(1:npop), :]
        for i = 1:npop
            if r < 0.5
                RB = rand(dim)
                X2 = Bast_P .+ (1 - t / max_iter)^2 .* (2 * RB .- 1) .* X[i, :]
                X2 = clamp.(X2, lb, ub)
            else
                K = round(Int, 1 + rand())
                R2 = rand(dim)
                X2 = X[i, :] .+ R2 .* (Xrandom .- K .* X[i, :])
                X2 = clamp.(X2, lb, ub)
            end

            f_newP2 = objfun(X2)
            if f_newP2 <= fit[i]
                X[i, :] = X2
                fit[i] = f_newP2
            end
        end

        SBOA_curve[t] = fbest
    end

    Best_score = fbest
    Best_pos = Bast_P
    # return Best_score, Best_pos, SBOA_curve
    return OptimizationResult(
        Best_pos,
        Best_score,
        SBOA_curve)
end

function SBOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SBOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end