"""
# References:

- Shehadeh, Hisham A. "Chernobyl disaster optimizer (CDO): A novel meta-heuristic method for global optimization."  Neural Computing and Applications 35, no. 15 (2023): 10733-10749.

"""
function CDO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return CDO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter) 
end

function CDO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Alpha_pos = zeros(Float64, dim)
    Alpha_score = Inf

    Beta_pos = zeros(Float64, dim)
    Beta_score = Inf

    Gamma_pos = zeros(Float64, dim)
    Gamma_score = Inf

    Positions = initialization(npop, dim, ub, lb)

    Convergence_curve = zeros(Float64, max_iter)

    l = 0

    while l < max_iter
        for i = 1:axes(Positions, 1)
            Flag4ub = Positions[i, :] .> ub
            Flag4lb = Positions[i, :] .< lb
            Positions[i, :] .= Positions[i, :] .* (.~(Flag4ub .+ Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness = objfun(Positions[i, :])

            if fitness < Alpha_score
                Alpha_score = fitness
                Alpha_pos = Positions[i, :]
            elseif fitness > Alpha_score && fitness < Beta_score
                Beta_score = fitness
                Beta_pos = Positions[i, :]
            elseif fitness > Alpha_score && fitness > Beta_score && fitness < Gamma_score
                Gamma_score = fitness
                Gamma_pos = Positions[i, :]
            end
        end

        a = 3 - l * (3 / max_iter)

        a1 = log10((16000 - 1) * rand() + 16000)
        a2 = log10((270000 - 1) * rand() + 270000)
        a3 = log10((300000 - 1) * rand() + 300000)

        for i = 1:axes(Positions, 1)
            for j = 1:axes(Positions, 2)
                r1, r2 = rand(), rand()
                pa = pi * r1^2 / (0.25 * a1) - a * rand()
                C1 = r2^2 * pi
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                va = 0.25 * (Alpha_pos[j] - pa * D_alpha)

                r1, r2 = rand(), rand()
                pb = pi * r1^2 / (0.5 * a2) - a * rand()
                C2 = r2^2 * pi
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                vb = 0.5 * (Beta_pos[j] - pb * D_beta)

                r1, r2 = rand(), rand()
                py = pi * r1^2 / a3 - a * rand()
                C3 = r2^2 * pi
                D_gamma = abs(C3 * Gamma_pos[j] - Positions[i, j])
                vy = Gamma_pos[j] - py * D_gamma

                Positions[i, j] = (va + vb + vy) / 3
            end
        end

        l += 1
        Convergence_curve[l] = Alpha_score
    end

    # return Alpha_score, Alpha_pos, Convergence_curve
    return OptimizationResult(
        Alpha_pos,
        Alpha_score,
        Convergence_curve)
end

function CDO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return CDO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end