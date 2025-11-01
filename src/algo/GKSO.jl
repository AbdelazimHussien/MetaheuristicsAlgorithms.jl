"""
# References:

- Hu, Gang, Yuxuan Guo, Guo Wei, and Laith Abualigah.
  "Genghis Khan shark optimizer: a novel nature-inspired algorithm for engineering optimization."
  Advanced Engineering Informatics 58 (2023): 102210.
"""
function GKSO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GKSO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GKSO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    lb = ones(dim) .* lb
    ub = ones(dim) .* ub
    VarSize = (dim)

    PopPos = zeros(npop, dim)
    PopFit = zeros(npop)
    PopPos[i, :] = initialization(npop, dim, ub, lb)
    for i = 1:npop
        # PopPos[i, :] = rand(dim) .* (ub - lb) + lb
        PopFit[i] = objfun(vec(PopPos[i, :]'))
    end

    Best_score = Inf
    Best_pos = []

    for i = 1:npop
        if PopFit[i] <= Best_score
            Best_score = PopFit[i]
            Best_pos = PopPos[i, :]
        end
    end

    curve = zeros(max_iter)
    h = [0.1]

    for it = 1:max_iter
        h = append!(h, 1 - 2 * h[end]^4)
        p = 2 * (1 - (it / max_iter)^(1 / 4)) + abs(h[end]) * ((it / max_iter)^(1 / 4) - (it / max_iter)^3)
        beta = 0.2 + (1.2 - 0.2) * (1 - (it / max_iter)^3)^2
        alpha = abs(beta * sin((3 * π / 2 + sin(3 * π / 2 * beta))))

        newPopPos = zeros(npop, dim)
        newPopFit = zeros(npop)
        for i = 1:npop
            newPopPos[i, :] = PopPos[i, :] + (lb + rand() * (ub - lb)) / it
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            newPopFit[i] = objfun(vec(newPopPos[i, :]'))
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        GKS_Pos = zeros(npop, dim)
        for i = 1:npop
            s = (1.5 * PopFit[i]^rand())
            s = real(s)
            if i == 1
                newPopPos[i, :] = (Best_pos - PopPos[i, :]) * s
            else
                GKS_Pos[i, :] = (Best_pos - PopPos[i, :]) * s
                newPopPos[i, :] = (GKS_Pos[i, :] + newPopPos[i-1, :]) / 2
            end
            if all(newPopPos[i, :] .>= lb) && all(newPopPos[i, :] .<= ub)
                newPopFit[i] = objfun(vec(newPopPos[i, :]'))
                if newPopFit[i] < PopFit[i]
                    PopPos[i, :] = newPopPos[i, :]
                    PopFit[i] = newPopFit[i]
                end
            end
        end

        for i = 1:npop
            TF = (rand() > 0.5) * 2 - 1
            newPopPos[i, :] = Best_pos + rand(dim) .* (Best_pos - PopPos[i, :]) + TF * p^2 * (Best_pos - PopPos[i, :])
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            newPopFit[i] = objfun(vec(newPopPos[i, :]'))
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        for i = 1:npop
            A1 = rand(1:npop, npop)
            r1, r2 = A1[1], A1[2]
            k = rand(1:npop)
            if rand() < 0.5
                k = rand(1:npop)
                f1, f2 = -1 + 2 * rand(), -1 + 2 * rand()
                ro = alpha * (2 * rand() - 1)
                Xk = rand(dim) .* (ub - lb) + lb

                L1 = rand() < 0.5
                u1 = L1 * 2 * rand() + (1 - L1)
                u2 = L1 * rand() + (1 - L1)
                u3 = L1 * rand() + (1 - L1)
                L2 = rand() < 0.5
                Xp = (1 - L2) * PopPos[k, :] + L2 * Xk
                popi1 = rand(VarSize) .* (ub - lb) + lb
                popi2 = rand(VarSize) .* (ub - lb) + lb

                A1 = rand(1:npop, 2)
                r1, r2 = A1[1], A1[2]
                k = rand(1:npop)

                if u1 < 0.5
                    newPopPos[i, :] = newPopPos[i, :] + f1 * (u1 * Best_pos - u2 * Xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (PopPos[r1, :] - PopPos[r2, :])) / 2
                else
                    newPopPos[i, :] = Best_pos + f1 * (u1 * Best_pos - u2 * Xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (PopPos[r1, :] - PopPos[r2, :])) / 2
                end
            end
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            newPopFit[i] = objfun(vec(newPopPos[i, :]'))
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        for i = 1:npop
            if PopFit[i] < Best_score
                Best_score = PopFit[i]
                Best_pos = PopPos[i, :]
            end
        end

        curve[it] = Best_score
    end

    # return Best_score, Best_pos, curve
    return OptimizationResult(
        Best_pos,
        Best_score,
        curve)
end

function GKSO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GKSO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end