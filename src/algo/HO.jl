"""
# References:

- Amiri, M.H., Mehrabi Hashjin, N., Montazeri, M., Mirjalili, S. and Khodadadi, N., 2024.
  Hippopotamus optimization algorithm: a novel nature-inspired optimization algorithm.
  Scientific Reports, 14(1), p.5032.
"""
function HO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return HO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end
function HO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    lb = fill(lb, dim)
    ub = fill(ub, dim)

    # Initialization
    X = initialization(npop, dim, lb, ub)
    fit = zeros(npop)
    for i = 1:npop
        fit[i] = objfun(X[i, :])
    end
    Xbest, fbest = nothing, Inf

    # Main Loop
    best_so_far = zeros(max_iter)
    for t = 1:max_iter
        best, location = findmin(fit)
        if t == 1 || best < fbest
            Xbest = X[location, :]
            fbest = best
        end

        for i = 1:(npop÷2)
            Dominant_hippopotamus = Xbest
            I1, I2 = rand(Bool) ? 1 : 2, rand(Bool) ? 1 : 2
            Ip1 = rand(Bool, 2)
            RandGroupNumber = rand(1:npop)
            RandGroup = sample(1:npop, RandGroupNumber; replace=false)

            MeanGroup = mean(X[RandGroup, :], dims=1)

            if length(RandGroup) != 1
                MeanGroup = mean(X[RandGroup, :], dims=1)
            else
                MeanGroup = X[RandGroup[1], :]
            end

            Alfa = [
                I2 * rand(dim) .+ .~Ip1[1],
                2 * rand(dim) .- 1,
                rand(dim),
                I1 * rand(dim) .+ .~Ip1[2],
                rand(),
            ]

            A, B = Alfa[rand(1:5)], Alfa[rand(1:5)]
            X_P1 = X[i, :] .+ rand() .* (Dominant_hippopotamus .- I1 .* X[i, :])
            T = exp(-t / max_iter)

            MeanGroup = vec(MeanGroup)

            if T > 0.6
                X_P2 = X[i, :] .+ A .* (Dominant_hippopotamus .- I2 .* MeanGroup)
            else
                if rand() > 0.5
                    X_P2 = X[i, :] .+ B .* (MeanGroup .- Dominant_hippopotamus)
                else
                    X_P2 = (ub .- lb) .* rand(dim) .+ lb
                end
            end
            X_P2 .= clamp.(X_P2, lb, ub)

            F_P1, F_P2 = objfun(X_P1), objfun(X_P2)
            if F_P1 < fit[i]
                X[i, :] = X_P1
                fit[i] = F_P1
            end
            if F_P2 < fit[i]
                X[i, :] = X_P2
                fit[i] = F_P2
            end
        end

        for i = (npop÷2+1):npop
            predator = lb .+ rand(dim) .* (ub .- lb)
            F_HL = objfun(predator)
            distance2Leader = abs.(predator .- X[i, :])

            b, c, d, l = rand(2.0:4.0), rand(1.0:1.5), rand(2.0:3.0), rand(-2*π:2*π)
            RL = 0.05 * levy(npop, dim, 1.5)

            if fit[i] > F_HL
                X_P3 = RL[i, :] .* predator .+ (b / (c - d * cos(l))) .* (1 ./ distance2Leader)
            else
                X_P3 = RL[i, :] .* predator .+ (b / (c - d * cos(l))) .* (1 ./ (2 .* distance2Leader .+ rand(dim)))
            end
            X_P3 .= clamp.(X_P3, lb, ub)

            F_P3 = objfun(X_P3)
            if F_P3 < fit[i]
                X[i, :] = X_P3
                fit[i] = F_P3
            end
        end

        for i = 1:npop
            LO_LOCAL = lb ./ t
            HI_LOCAL = ub ./ t
            D = [rand(dim) * 2 .- 1, rand(), randn()][rand(1:3)]
            X_P4 = X[i, :] .+ rand() .* (LO_LOCAL .+ D .* (HI_LOCAL .- LO_LOCAL))
            X_P4 .= clamp.(X_P4, lb, ub)

            F_P4 = objfun(X_P4)
            if F_P4 < fit[i]
                X[i, :] = X_P4
                fit[i] = F_P4
            end
        end

        best_so_far[t] = fbest
    end

    # return fbest, Xbest, best_so_farx
    return OptimizationResult(
        Xbest,
        fbest,
        best_so_farx)
end

function HO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return HO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end