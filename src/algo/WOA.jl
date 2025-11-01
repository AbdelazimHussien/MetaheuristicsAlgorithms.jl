"""
# References:

- Mirjalili, Seyedali, and Andrew Lewis.
  "The whale optimization algorithm."
  Advances in engineering software 95 (2016): 51-67.
"""
function WOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return WOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

# function WOA(npop::Integer, max_iter::Integer, lb::Vector{Float64}, ub::Vector{Float64}, dim::Integer, objfun)
function WOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)

    leader_pos = zeros(dim)
    leader_score = Inf

    Positions = initialization(npop, dim, ub, lb)

    convergence_curve = zeros(max_iter, 1)

    t = 0

    while t < max_iter
        for i in axes(Positions, 1)
            Flag4ub = Positions[i, :] .> ub
            Flag4lb = Positions[i, :] .< lb
            Positions[i, :] .= Positions[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness = objfun(Positions[i, :])

            if fitness < leader_score
                leader_score = fitness
                # leader_pos = Positions[i, :]
                leader_pos = vec(Positions[i, :])
            end
        end

        a = 2 - t * (2 / max_iter)
        a2 = -1 + t * (-1 / max_iter)

        for i in axes(Positions, 1)
            r1 = rand()
            r2 = rand()

            A = 2 * a * r1 - a
            C = 2 * r2

            b = 1
            l = (a2 - 1) * rand() + 1

            p = rand()

            for j in axes(Positions, 2)
                if p < 0.5
                    if abs(A) >= 1
                        rand_leader_index = floor(Int, npop * rand() + 1)
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand
                    elseif abs(A) < 1
                        D_Leader = abs(C * leader_pos[j] - Positions[i, j])
                        Positions[i, j] = leader_pos[j] - A * D_Leader
                    end
                elseif p >= 0.5
                    distance2Leader = abs(leader_pos[j] - Positions[i, j])
                    Positions[i, j] = distance2Leader * exp(b * l) * cos(l * 2 * pi) + leader_pos[j]
                end
            end
        end
        t += 1
        convergence_curve[t] = leader_score
    end

    # return leader_score, leader_pos, convergence_curve
    return OptimizationResult(
        leader_pos,
        leader_score,
        convergence_curve)
end

# function WOA(problem::OptimizationProblem, npop::Integer, max_iter::Integer)
#     dim = problem.dim
#     objfun = problem.objfun
#     lb = problem.lb
#     ub = problem.ub
#     return WOA(npop, max_iter, lb, ub, dim, objfun)
# end

function WOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)
    return WOA(npop, max_iter, problem.lb, problem.ub, problem.dim, problem.objfun)
end