"""
# References:

- Mirjalili, Seyedali.
  "SCA: a sine cosine algorithm for solving optimization problems."
  Knowledge-based systems 96 (2016): 120-133.
"""
function SCA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SCA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SCA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    println("SCA is optimizing your problem")

    X = initialization(npop, dim, ub, lb)

    Destination_position = zeros(dim)
    Destination_fitness = Inf

    Convergence_curve = zeros(max_iter)
    Objective_values = zeros(size(X, 1))

    for i in axes(X, 1)
        Objective_values[i] = objfun(X[i, :])
        if i == 1
            Destination_position .= X[i, :]
            Destination_fitness = Objective_values[i]
        elseif Objective_values[i] < Destination_fitness
            Destination_position .= X[i, :]
            Destination_fitness = Objective_values[i]
        end
    end

    t = 2
    while t <= max_iter
        a = 2
        r1 = a - t * (a / max_iter)

        for i in axes(X, 1)
            for j in axes(X, 2)
                r2 = 2 * pi * rand()
                r3 = 2 * rand()
                r4 = rand()

                if r4 < 0.5
                    X[i, j] += r1 * sin(r2) * abs(r3 * Destination_position[j] - X[i, j])
                else
                    X[i, j] += r1 * cos(r2) * abs(r3 * Destination_position[j] - X[i, j])
                end
            end
        end

        for i in axes(X, 1)
            X[i, :] = clamp.(X[i, :], lb, ub)

            Objective_values[i] = objfun(X[i, :])

            if Objective_values[i] < Destination_fitness
                Destination_position .= X[i, :]
                Destination_fitness = Objective_values[i]
            end
        end

        Convergence_curve[t] = Destination_fitness
        println("$t --> ", Convergence_curve[t])

        t += 1
    end

    # return Destination_fitness, Destination_position, Convergence_curve
    return OptimizationResult(
        Destination_position,
        Destination_fitness,
        Convergence_curve)
end

function SCA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SCA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end