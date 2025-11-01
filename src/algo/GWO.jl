"""
# References:

- Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis.
  "Grey wolf optimizer."
  Advances in engineering software 69 (2014): 46-61.
"""
function GWO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GWO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GWO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # Alpha_pos = zeros(1, dim)
    Alpha_pos = zeros(dim)
    Alpha_score = Inf

    # Beta_pos = zeros(1, dim)
    Beta_pos = zeros(dim)
    Beta_score = Inf

    # Delta_pos = zeros(1, dim)
    Delta_pos = zeros(dim)
    Delta_score = Inf

    Positions = initialization(npop, dim, ub, lb)

    Convergence_curve = zeros(max_iter)

    l = 0
    while l < max_iter
        for i in axes(Positions, 1)
            Positions[i, :] = max.(min.(Positions[i, :], ub), lb)

            fitness = objfun(Positions[i, :])

            if fitness < Alpha_score
                Alpha_score = fitness
                Alpha_pos = Positions[i, :]
            end

            if fitness > Alpha_score && fitness < Beta_score
                Beta_score = fitness
                Beta_pos = Positions[i, :]
            end

            if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
                Delta_score = fitness
                Delta_pos = Positions[i, :]
            end
        end
        a = 2 - l * (2 / max_iter)

        for i in axes(Positions, 1)
            for j in axes(Positions, 2)
                r1 = rand()
                r2 = rand()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = rand()
                r2 = rand()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = rand()
                r2 = rand()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3
            end
        end

        l += 1
        Convergence_curve[l] = Alpha_score
    end
    # return Alpha_score, Alpha_pos, Convergence_curve
    return OptimizationResult(
        Alpha_pos,
        Alpha_score,
        vec(Convergence_curve))
end

function GWO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GWO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end