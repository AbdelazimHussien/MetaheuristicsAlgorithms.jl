"""
# References:

Dehghani, M., Montazeri, Z., Trojovská, E., & Trojovský, P. "Coati Optimization Algorithm: A new bio-inspired metaheuristic algorithm for solving optimization problems." Knowledge-based systems 259 (2023): 110011.
"""
function CoatiOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return CoatiOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter) 
end

function CoatiOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    lb = ones(dim) .* lb
    ub = ones(dim) .* ub

    # INITIALIZATION
    X = zeros(npop, dim)
    X = initialization(npop, dim, ub, lb)
    # for i = 1:dim
    #     X[:, i] = lb[i] .+ rand(npop) .* (ub[i] - lb[i])
    # end

    fit = [objfun(X[i, :]) for i = 1:npop]

    Xbest = zeros(dim)
    fbest = Inf

    best_so_far = 0

    for t = 1:max_iter
        best, location = findmin(fit)
        if t == 1 || best < fbest
            fbest = best
            Xbest = X[location, :]
        end

        X_P1 = zeros(npop, dim)
        F_P1 = zeros(npop)
        for i = 1:div(npop, 2)
            iguana = Xbest
            I = round(Int, 1 + rand())
            X_P1[i, :] = X[i, :] .+ rand() .* (iguana - I .* X[i, :])  # Eq. (4)
            X_P1[i, :] = clamp.(X_P1[i, :], lb, ub)

            L = X_P1[i, :]
            F_P1[i] = objfun(L)
            if F_P1[i] < fit[i]
                X[i, :] = X_P1[i, :]
                fit[i] = F_P1[i]
            end
        end

        for i = div(npop, 2)+1:npop
            iguana = lb .+ rand(dim) .* (ub - lb)  # Eq (5)
            F_HL = objfun(iguana)
            I = round(Int, 1 + rand())

            if fit[i] > F_HL
                X_P1[i, :] = X[i, :] .+ rand() .* (iguana - I .* X[i, :])  # Eq. (6)
            else
                X_P1[i, :] = X[i, :] .+ rand() .* (X[i, :] - iguana)  # Eq. (6)
            end
            X_P1[i, :] = clamp.(X_P1[i, :], lb, ub)

            L = X_P1[i, :]
            F_P1[i] = objfun(L)
            if F_P1[i] < fit[i]
                X[i, :] = X_P1[i, :]
                fit[i] = F_P1[i]
            end
        end

        X_P2 = zeros(npop, dim)
        F_P2 = zeros(npop)
        for i = 1:npop
            LO_LOCAL = lb ./ t
            HI_LOCAL = ub ./ t

            X_P2[i, :] = X[i, :] .+ (1 - 2 .* rand()) .* (LO_LOCAL .+ rand() .* (HI_LOCAL - LO_LOCAL))  # Eq. (8)
            X_P2[i, :] = clamp.(X_P2[i, :], LO_LOCAL, HI_LOCAL)

            L = X_P2[i, :]
            F_P2[i] = objfun(L)
            if F_P2[i] < fit[i]
                X[i, :] = X_P2[i, :]
                fit[i] = F_P2[i]
            end
        end

        best_so_far = fbest
        average = mean(fit)
    end

    # return fbest, Xbest, best_so_far
    return OptimizationResult(
        Xbest,
        fbest,
        best_so_far)
end

function CoatiOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return CoatiOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end