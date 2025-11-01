"""
# References:

- Abdollahzadeh, Benyamin, Farhad Soleimanian Gharehchopogh, and Seyedali Mirjalili.
  "Artificial gorilla troops optimizer: a new nature‐inspired metaheuristic algorithm for global optimization problems."
  International Journal of Intelligent Systems 36, no. 10 (2021): 5887-5958.
"""
function GTO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GTO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GTO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Silverback = []
    Silverback_Score = Inf

    X = initialization(npop, dim, ub, lb)

    convergence_curve = zeros(max_iter)

    Pop_Fit = zeros(npop)
    for i = 1:npop
        Pop_Fit[i] = objfun(X[i, :])
        if Pop_Fit[i] < Silverback_Score
            Silverback_Score = Pop_Fit[i]
            Silverback = X[i, :]
        end
    end

    GX = copy(X)
    lb = ones(dim) * lb
    ub = ones(dim) * ub

    p = 0.03
    Beta = 3
    w = 0.8

    for It = 1:max_iter
        a = (cos(2 * rand()) + 1) * (1 - It / max_iter)
        C = a * (2 * rand() - 1)

        for i = 1:npop
            if rand() < p
                GX[i, :] .= (ub - lb) .* rand() .+ lb
            else
                if rand() >= 0.5
                    Z = rand(dim) * (2 * a) .- a
                    H = Z .* X[i, :]
                    GX[i, :] .= (rand() - a) * X[rand(1:npop), :] .+ C .* H
                else
                    GX[i, :] .= X[i, :] .- C .* (C * (X[i, :] .- GX[rand(1:npop), :]) .+ rand() * (X[i, :] .- GX[rand(1:npop), :]))
                end
            end
        end

        GX .= clamp.(GX, lb, ub)

        for i = 1:npop
            New_Fit = objfun(GX[i, :])
            if New_Fit < Pop_Fit[i]
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            end
            if New_Fit < Silverback_Score
                Silverback_Score = New_Fit
                Silverback = GX[i, :]
            end
        end

        for i = 1:npop
            if a >= w
                g = 2^C
                delta = (abs.(mean(GX, dims=1)) .^ g) .^ (1 / g)
                GX[i, :] .= C * delta[1, :] .* (X[i, :] .- Silverback) .+ X[i, :]
            else
                h = rand() >= 0.5 ? randn() : randn()
                r1 = rand()
                GX[i, :] .= Silverback .- (Silverback .* (2 * r1 - 1) .- X[i, :] .* (2 * r1 - 1)) .* (Beta * h)
            end
        end

        GX .= clamp.(GX, lb, ub)

        for i = 1:npop
            New_Fit = objfun(GX[i, :])
            if New_Fit < Pop_Fit[i]
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            end
            if New_Fit < Silverback_Score
                Silverback_Score = New_Fit
                Silverback = GX[i, :]
            end
        end

        convergence_curve[It] = Silverback_Score
    end

    # return Silverback_Score, Silverback, convergence_curve
    return OptimizationResult(
        Silverback,
        Silverback_Score,
        convergence_curve)
end

function GTO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GTO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end