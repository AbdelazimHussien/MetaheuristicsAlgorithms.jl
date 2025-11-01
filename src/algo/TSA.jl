"""
# References:

- Kaur, Satnam, Lalit K. Awasthi, Amrit Lal Sangal, and Gaurav Dhiman.
  "Tunicate Swarm Algorithm: A new bio-inspired based metaheuristic paradigm for global optimization."
  Engineering Applications of Artificial Intelligence 90 (2020): 103541.
"""
function TSA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return TSA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function TSA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    score = Inf
    position = zeros(dim)
    positions = initialization(npop, dim, ub, lb)
    convergence = zeros(max_iter)

    t = 0

    while t < max_iter
        for i in axes(positions, 1)
            Flag4ub = positions[i, :] .> ub
            Flag4lu = positions[i, :] .< lb
            positions[i, :] = (positions[i, :] .* .~(Flag4ub .+ Flag4lu)) .+ ub .* Flag4ub .+ lb .* Flag4lu

            fitness = objfun(positions[i, :])

            if fitness < score
                score = fitness
                position = copy(positions[i, :])
            end
        end

        xmin = 1
        xmax = 4
        xr = xmin + rand() * (xmax - xmin)
        xr = floor(Int, xr)

        for i in axes(positions, 1)
            for j in axes(positions, 2)
                A1 = ((rand() + rand()) - (2 * rand())) / xr
                c2 = rand()

                if i == 1
                    c3 = rand()
                    d_pos = abs(position[j] - c2 * positions[i, j])

                    if c3 >= 0
                        positions[i, j] = position[j] + A1 * d_pos
                    else
                        positions[i, j] = position[j] - A1 * d_pos
                    end
                else
                    c3 = rand()
                    d_pos = abs(position[j] - c2 * positions[i, j])

                    if c3 >= 0
                        Pos_i_j = position[j] + A1 * d_pos
                    else
                        Pos_i_j = position[j] - A1 * d_pos
                    end

                    positions[i, j] = (Pos_i_j + positions[i-1, j]) / 2
                end
            end
        end

        t += 1
        convergence[t] = score
    end

    # return score, position, convergence
    return OptimizationResult(
        position,
        score,
        convergence)
end

function TSA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return TSA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end