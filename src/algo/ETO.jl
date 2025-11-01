"""
# References:

- Luan, Tran Minh, Samir Khatir, Minh Thi Tran, Bernard De Baets, and Thanh Cuong-Le.
  "Exponential-trigonometric optimization algorithm for solving complicated engineering problems."
  Computer Methods in Applied Mechanics and Engineering 432 (2024): 117411.
"""
function ETO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return ETO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function ETO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Destination_position = zeros(dim)
    Destination_fitness = Inf
    Destination_position_second = zeros(dim)
    Convergence_curve = zeros(max_iter)
    Position_sort = zeros(npop, dim)

    b = 1.55
    CE = floor(Int, 1 + max_iter / b)
    T = floor(Int, 1.2 + max_iter / 2.25)
    CEi = 0
    CEi_temp = 0
    UB_2 = ub
    LB_2 = lb

    X = initialization(npop, dim, ub, lb)
    Objective_values = zeros(npop)

    for i = 1:npop
        Objective_values[i] = objfun(X[i, :])
        if Objective_values[i] < Destination_fitness
            Destination_position .= X[i, :]
            Destination_fitness = Objective_values[i]
        end
    end
    Convergence_curve[1] = Destination_fitness
    t = 2

    while t <= max_iter
        for i = 1:npop
            for j = 1:dim
                d1 = 0.1 * exp(-0.01 * t) * cos(0.5 * max_iter * (1 - t / max_iter))
                d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * max_iter * (1 - t / max_iter))

                CM = (sqrt(t / max_iter)^tan(d1 / d2)) * rand() * 0.01

                if t == CEi
                    UB_2 = Destination_position[j] + (1 - t / max_iter) * abs(rand() * Destination_position[j] - Destination_position_second[j]) * rand()
                    LB_2 = Destination_position[j] - (1 - t / max_iter) * abs(rand() * Destination_position[j] - Destination_position_second[j]) * rand()

                    if UB_2 > ub
                        UB_2 = ub
                    end
                    if LB_2 < lb
                        LB_2 = lb
                    end

                    X = initialization(npop, dim, UB_2, LB_2)
                    CEi_temp = CEi
                    CEi = 0
                end

                if t <= T
                    q1, q3, q4, q5 = rand(), rand(), rand(), rand()

                    if CM > 1
                        d1 = 0.1 * exp(-0.01 * t) * cos(0.5 * max_iter * q1)
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * max_iter * q1)
                        alpha_1 = q5 * 3 * (t / max_iter - 0.85) * exp(abs(d1 / d2) - 1)

                        if q1 <= 0.5
                            X[i, j] = Destination_position[j] + q5 * alpha_1 * abs(Destination_position[j] - X[i, j])
                        else
                            X[i, j] = Destination_position[j] - q5 * alpha_1 * abs(Destination_position[j] - X[i, j])
                        end
                    else
                        d1 = 0.1 * exp(-0.01 * t) * cos(0.5 * max_iter * q3)
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * max_iter * q3)
                        alpha_3 = rand() * 3 * (t / max_iter - 0.85) * exp(abs(d1 / d2) - 1.3)

                        if q3 <= 0.5
                            X[i, j] = Destination_position[j] + q4 * alpha_3 * abs(q5 * Destination_position[j] - X[i, j])
                        else
                            X[i, j] = Destination_position[j] - q4 * alpha_3 * abs(q5 * Destination_position[j] - X[i, j])
                        end
                    end
                else
                    q2, q6 = rand(), rand()
                    alpha_2 = q6 * exp(tanh(1.5 * (-t / max_iter - 0.75) - q6))

                    if CM < 1
                        d1 = 0.1 * exp(-0.01 * t) * cos(0.5 * max_iter * q2)
                        d2 = -0.1 * exp(-0.01 * t) * cos(0.5 * max_iter * q2)
                        X[i, j] += exp(tan(abs(d1 / d2)) * abs(q6 * alpha_2 * Destination_position[j] - X[i, j]))
                    else
                        if q2 <= 0.5
                            X[i, j] += 3 * abs(q6 * alpha_2 * Destination_position[j] - X[i, j])
                        else
                            X[i, j] -= 3 * abs(q6 * alpha_2 * Destination_position[j] - X[i, j])
                        end
                    end
                end
            end
            CEi = CEi_temp
        end

        for i = 1:npop
            X[i, :] = clamp.(X[i, :], LB_2, UB_2)
            Objective_values[i] = objfun(X[i, :])

            if Objective_values[i] < Destination_fitness
                Destination_position .= X[i, :]
                Destination_fitness = Objective_values[i]
            end
        end

        if t == CE
            CEi = CE + 1
            CE += floor(Int, 2 - t * 2 / (max_iter - CE * 4.6) / 1)
            temp = zeros(dim)
            temp2 = zeros(npop, dim)

            for i = 1:(npop-1)
                for j = 1:(npop-1-i)
                    if Objective_values[j] > Objective_values[j+1]
                        temp .= Objective_values[j]
                        Objective_values[j] = Objective_values[j+1]
                        Objective_values[j+1] = temp[1]
                        temp2[j, :] .= Position_sort[j, :]
                        Position_sort[j, :] .= Position_sort[j+1, :]
                        Position_sort[j+1, :] .= temp2[j, :]
                    end
                end
            end
            Destination_position_second .= Position_sort[2, :]
        end

        Convergence_curve[t] = Destination_fitness
        t += 1
    end

    # return Destination_fitness, Destination_position, Convergence_curve
    return OptimizationResult(
        Destination_position,
        Destination_fitness,
        Convergence_curve)
end

function ETO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return ETO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end