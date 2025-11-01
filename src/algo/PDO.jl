
# function levym(n, m, beta)
#     num = gamma(1 + beta) * sin(pi * beta / 2)
#     den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)
#     sigma_u = (num / den)^(1 / beta)

#     u = rand(Normal(0, sigma_u), n, m)
#     v = rand(Normal(0, 1), n, m)

#     z = u ./ (abs.(v) .^ (1 / beta))
#     return z
# end

"""
# References:

- Ezugwu, Absalom E., Jeffrey O. Agushaka, Laith Abualigah, Seyedali Mirjalili, and Amir H. Gandomi.
  "Prairie dog optimization algorithm."
  Neural Computing and Applications 34, no. 22 (2022): 20017-20065.
"""
function PDO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return PDO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function PDO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    PDBest_P = zeros(dim)
    Best_PD = Inf
    X = initialization(npop, dim, ub, lb)
    Xnew = zeros(npop, dim)
    PDConv = zeros(max_iter)
    M = dim
    t = 1
    rho = 0.005
    epsPD = 0.1
    OBest = zeros(npop)
    CBest = zeros(npop)

    for i = 1:npop
        Flag_UB = X[i, :] .> ub
        Flag_LB = X[i, :] .< lb
        X[i, :] .= (X[i, :] .* .~(Flag_UB .| Flag_LB)) .+ ub .* Flag_UB .+ lb .* Flag_LB

        OBest[i] = objfun(X[i, :])
        if OBest[i] < Best_PD
            Best_PD = OBest[i]
            PDBest_P .= X[i, :]
        end
    end

    while t <= max_iter
        mu = ifelse(t % 2 == 0, -1, 1)
        DS = 1.5 * randn() * (1 - t / max_iter)^(2 * t / max_iter) * mu
        PE = 1.5 * (1 - t / max_iter)^(2 * t / max_iter) * mu
        RL = levy(npop, dim, 1.5)
        TPD = repeat(PDBest_P', npop)

        for i = 1:npop
            for j = 1:M
                cpd = rand() * ((TPD[i, j] - X[rand(1:npop), j]) / (TPD[i, j] + eps()))
                P = rho + (X[i, j] - mean(X[i, :])) / (TPD[i, j] * (ub - lb) + eps())
                eCB = PDBest_P[j] * P

                if t < max_iter / 4
                    Xnew[i, j] = PDBest_P[j] - eCB * epsPD - cpd * RL[i, j]
                elseif t < 2 * max_iter / 4
                    Xnew[i, j] = PDBest_P[j] * X[rand(1:npop), j] * DS * RL[i, j]
                elseif t < 3 * max_iter / 4
                    Xnew[i, j] = PDBest_P[j] * PE * rand()
                else
                    Xnew[i, j] = PDBest_P[j] - eCB * eps() - cpd * rand()
                end
            end

            Flag_UB = Xnew[i, :] .> ub
            Flag_LB = Xnew[i, :] .< lb
            Xnew[i, :] .= (Xnew[i, :] .* .~(Flag_UB .| Flag_LB)) .+ ub .* Flag_UB .+ lb .* Flag_LB

            CBest[i] = objfun(Xnew[i, :])
            if CBest[i] < OBest[i]
                X[i, :] .= Xnew[i, :]
                OBest[i] = CBest[i]
            end
            if OBest[i] < Best_PD
                Best_PD = OBest[i]
                PDBest_P .= X[i, :]
            end
        end

        PDConv[t] = Best_PD

        t += 1
    end

    # return Best_PD, PDBest_P, PDConv
    return OptimizationResult(
        PDBest_P,
        Best_PD,
        PDConv)
end

function PDO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return PDO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end