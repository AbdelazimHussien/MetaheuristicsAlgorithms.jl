"""
# References:

- Chopra, Nitish, and Muhammad Mohsin Ansari.
  "Golden jackal optimization: A novel nature-inspired optimizer for engineering applications."
  Expert Systems with Applications 198 (2022): 116924.
"""
function GJO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GJO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GJO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Male_Jackal_pos = zeros(1, dim)
    Male_Jackal_score = Inf
    Female_Jackal_pos = zeros(1, dim)
    Female_Jackal_score = Inf

    Positions = initialization(npop, dim, ub, lb)

    Convergence_curve = zeros(max_iter)

    l = 0

    while l < max_iter
        for i in axes(Positions, 1)
            Flag4ub = Positions[i, :] .> ub
            Flag4lb = Positions[i, :] .< lb
            Positions[i, :] = min.(max.(Positions[i, :], lb), ub)

            fitness = objfun(Positions[i, :])

            if fitness < Male_Jackal_score
                Male_Jackal_score = fitness
                Male_Jackal_pos = Positions[i, :]
            end
            if fitness > Male_Jackal_score && fitness < Female_Jackal_score
                Female_Jackal_score = fitness
                Female_Jackal_pos = Positions[i, :]
            end
        end

        E1 = 1.5 * (1 - (l / max_iter))
        RL = 0.05 * levy(npop, dim, 1.5)

        Male_Positions = zeros(npop, dim)
        Female_Positions = zeros(npop, dim)

        for i in axes(Positions, 1)
            for j in axes(Positions, 2)
                r1 = rand()
                E0 = 2 * r1 - 1
                E = E1 * E0

                if abs(E) < 1
                    D_male_jackal = abs((RL[i, j] * Male_Jackal_pos[j] - Positions[i, j]))
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((RL[i, j] * Female_Jackal_pos[j] - Positions[i, j]))
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal
                else
                    D_male_jackal = abs((Male_Jackal_pos[j] - RL[i, j] * Positions[i, j]))
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((Female_Jackal_pos[j] - RL[i, j] * Positions[i, j]))
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal
                end

                Positions[i, j] = (Male_Positions[i, j] + Female_Positions[i, j]) / 2
            end
        end

        l += 1
        Convergence_curve[l] = Male_Jackal_score
    end

    # return Male_Jackal_score, Male_Jackal_pos, Convergence_curve
    return OptimizationResult(
        Male_Jackal_pos,
        Male_Jackal_score,
        Convergence_curve)
end

function GJO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GJO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end