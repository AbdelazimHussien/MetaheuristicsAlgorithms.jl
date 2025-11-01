"""
# References:

- Ahmadianfar, Iman, Omid Bozorg-Haddad, and Xuefeng Chu.
  "Gradient-based optimizer: A new metaheuristic optimization algorithm."
  Information Sciences 540 (2020): 131-159.
"""
function GBO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GBO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GBO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    nV = dim
    pr = 0.5
    lb = ones(dim) .* lb
    ub = ones(dim) .* ub
    Cost = zeros(npop)
    X = initialization(npop, nV, ub, lb)
    Convergence_curve = zeros(max_iter)

    for i = 1:npop
        Cost[i] = objfun(X[i, :])
    end

    sort_idx = sortperm(Cost)
    Best_Cost = Cost[sort_idx[1]]
    Best_X = X[sort_idx[1], :]
    Worst_Cost = Cost[sort_idx[end]]
    Worst_X = X[sort_idx[end], :]

    for it = 1:max_iter
        beta = 0.2 + (1.2 - 0.2) * (1 - (it / max_iter)^3)^2
        alpha = abs(beta * sin(3 * π / 2 + sin(3 * π / 2 * beta)))

        for i = 1:npop
            A1 = rand(1:npop, 4)
            r1, r2, r3, r4 = A1

            Xm = (X[r1, :] + X[r2, :] + X[r3, :] + X[r4, :]) / 4
            ro = alpha * (2 * rand() - 1)
            ro1 = alpha * (2 * rand() - 1)
            eps = 5e-3 * rand()

            DM = rand() * ro * (Best_X - X[r1, :])
            Flag = 1
            GSR = GradientSearchRule(ro1, Best_X, Worst_X, X[i, :], X[r1, :], DM, eps, Xm, Flag)
            DM = rand() * ro * (Best_X - X[r1, :])
            X1 = X[i, :] - GSR + DM

            DM = rand() * ro * (X[r1, :] - X[r2, :])
            Flag = 2
            GSR = GradientSearchRule(ro1, Best_X, Worst_X, X[i, :], X[r1, :], DM, eps, Xm, Flag)
            DM = rand() * ro * (X[r1, :] - X[r2, :])
            X2 = Best_X - GSR + DM

            Xnew = zeros(nV)
            for j = 1:nV
                ro = alpha * (2 * rand() - 1)
                X3 = X[i, j] - ro * (X2[j] - X1[j])
                ra, rb = rand(), rand()
                Xnew[j] = ra * (rb * X1[j] + (1 - rb) * X2[j]) + (1 - ra) * X3
            end

            if rand() < pr
                k = rand(1:npop)
                f1 = -1 + 2 * rand()
                f2 = -1 + 2 * rand()
                ro = alpha * (2 * rand() - 1)
                Xk = lb .+ (ub - lb) .* rand(nV)

                L1 = rand() < 0.5
                u1 = L1 * 2 * rand() + (1 - L1)
                u2 = L1 * rand() + (1 - L1)
                u3 = L1 * rand() + (1 - L1)
                L2 = rand() < 0.5
                Xp = (1 - L2) * X[k, :] + L2 * Xk

                if u1 < 0.5
                    Xnew = Xnew + f1 * (u1 * Best_X - u2 * Xp) + f2 * ro * (u3 * (X2 - X1) + u2 * (X[r1, :] - X[r2, :])) / 2
                else
                    Xnew = Best_X + f1 * (u1 * Best_X - u2 * Xp) + f2 * ro * (u3 * (X2 - X1) + u2 * (X[r1, :] - X[r2, :])) / 2
                end
            end

            Flag4ub = Xnew .> ub
            Flag4lb = Xnew .< lb
            Xnew = (Xnew .* (Flag4ub .+ Flag4lb .== 0)) .+ ub .* Flag4ub .+ lb .* Flag4lb
            Xnew_Cost = objfun(Xnew)

            if Xnew_Cost < Cost[i]
                X[i, :] = Xnew
                Cost[i] = Xnew_Cost
                if Cost[i] < Best_Cost
                    Best_X = X[i, :]
                    Best_Cost = Cost[i]
                end
            end

            if Cost[i] > Worst_Cost
                Worst_X = X[i, :]
                Worst_Cost = Cost[i]
            end
        end

        Convergence_curve[it] = Best_Cost
    end

    # return Best_Cost, Best_X, Convergence_curve
    return OptimizationResult(
        Best_X,
        Best_Cost,
        Convergence_curve)
end

function GradientSearchRule(ro1, Best_X, Worst_X, X, Xr1, DM, eps, Xm, Flag)
    nV = length(X)
    Delta = 2 .* rand(nV) .* abs.(Xm - X)
    Step = ((Best_X - Xr1) + Delta) / 2
    DelX = rand(nV) .* abs.(Step)

    GSR = randn() .* ro1 .* (2 .* DelX .* X) ./ (Best_X - Worst_X .+ eps)
    if Flag == 1
        Xs = X - GSR + DM
    else
        Xs = Best_X - GSR + DM
    end

    yp = rand() .* (0.5 .* (Xs + X) .+ rand() .* DelX)
    yq = rand() .* (0.5 .* (Xs + X) .- rand() .* DelX)
    GSR = randn() .* ro1 .* (2 .* DelX .* X) ./ (yp - yq .+ eps)

    return GSR
end

function GBO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GBO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end