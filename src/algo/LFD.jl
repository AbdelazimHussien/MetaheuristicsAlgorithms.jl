"""
# References:

- Houssein, Essam H., Mohammed R. Saad, Fatma A. Hashim, Hassan Shaban, and M. Hassaballah.
  "Lévy flight distribution: A new metaheuristic algorithm for solving engineering optimization problems."
  Engineering Applications of Artificial Intelligence 94 (2020): 103731.
"""
function LFD(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return LFD(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function LFD(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    threshold = 2
    lb = lb * ones(dim)
    ub = ub * ones(dim)

    if size(ub, 1) == 1
        ub = ones(dim) * ub
        lb = ones(dim) * lb
    end

    Positions = initialization(npop, dim, ub, lb)

    PositionsFitness = zeros(npop)
    Positions_temp = copy(Positions)

    for i in axes(Positions, 1)
        PositionsFitness[i] = objfun(Positions[i, :])
    end

    sorted_fitness, sorted_indexes = findmin(PositionsFitness)
    TargetPosition = Positions[sorted_indexes, :]
    TargetFitness = sorted_fitness
    vec_flag = [1, -1]
    NN = zeros(Int, npop)

    conver_iter = zeros(max_iter)
    conver_iter[1] = TargetFitness
    l = 1

    D = zeros(npop)
    pos_temp_nei = Vector{Vector{Float64}}(undef, npop)

    while l <= max_iter
        for i in axes(Positions, 1)
            Positions[i, :] .= max.(lb[dim], min.(Positions[i, :], ub[dim]))

            S_i = zeros(dim)
            NeighborN = 0

            for j = 1:npop
                if i != j
                    dis = norm(Positions[j, :] - Positions[i, :])
                    if (dis < threshold)
                        temp = PositionsFitness[j] / (PositionsFitness[i] + eps())
                        temp = (0.9 * (temp - minimum(temp))) / (maximum(temp) - minimum(temp) + eps()) + 0.1
                        NeighborN += 1
                        D[NeighborN] = temp

                        pos_temp_nei[NeighborN] = Positions[j, :]

                        R = rand()
                        CSV = 0.5
                        if R < CSV
                            rand_leader_index = rand(1:npop)
                            X_rand = Positions[rand_leader_index, :]
                            Positions_temp[j, :] = levy(Positions[j, :], X_rand, dim)
                        else
                            Positions_temp[j, :] = lb[1] .+ rand(1, dim) .* (ub[1] - lb[1])
                        end
                    end
                end
            end

            for p = 1:NeighborN
                flag_index = rand(1:2)
                var_flag = vec_flag[flag_index]
                s_ij = var_flag * D[p] * (pos_temp_nei[p]) / NeighborN
                S_i .+= s_ij
            end

            S_i_total = S_i
            rand_leader_index = rand(1:floor(Int(npop)))
            X_rand = Positions[rand_leader_index, :]
            X_new = TargetPosition + 10 * S_i_total + rand() * 0.00005 * ((TargetPosition + 0.005 * X_rand) / 2 - Positions[i, :])
            X_new = levy(X_new, TargetPosition, dim)
            Positions_temp[i, :] = X_new
            NN[i] = NeighborN
        end

        Positions .= Positions_temp

        for i in axes(Positions, 1)
            PositionsFitness[i] = objfun(Positions[i, :])
        end

        xminn, x_pos_min = findmin(PositionsFitness)
        if xminn < TargetFitness
            TargetPosition = Positions[x_pos_min, :]
            TargetFitness = xminn
        end

        conver_iter[l] = TargetFitness
        l += 1
    end
    # return TargetFitness, TargetPosition, conver_iter
    return OptimizationResult(
        TargetPosition,
        TargetFitness,
        conver_iter)
end

function LFD(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return LFD(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

# function Levy(pos, Pos_target, dim)
#     beta = 3 / 2
#     sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)

#     for j = 1:dim
#         u = rand() * sigma
#         v = rand()
#         step = u / abs(v)^(1 / beta)
#         stepsize = 0.01 * step * (pos[j] - Pos_target[j])
#         pos[j] += stepsize * rand()
#     end

#     return pos
# end