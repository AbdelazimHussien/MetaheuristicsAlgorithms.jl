"""
# References:

- Bai, Jianfu, Yifei Li, Mingpo Zheng, Samir Khatir, Brahim Benaissa, Laith Abualigah, and Magd Abdel Wahab.
  "A sinh cosh optimizer."
  Knowledge-Based Systems 282 (2023): 111081.
"""
function SCHO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SCHO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SCHO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Destination_position = zeros(1, dim)
    Destination_fitness = Inf
    Destination_position_second = zeros(1, dim)
    Convergence_curve = zeros(max_iter)
    Position_sort = zeros(npop, dim)

    u = 0.388
    m = 0.45
    n = 0.5
    p = 10
    q = 9
    Alpha = 4.6
    Beta = 1.55
    BS = floor(Int, max_iter / Beta)
    ct = 3.6
    T = floor(Int, max_iter / ct)
    BSi = 0
    BSi_temp = 0
    ub_2 = ub
    lb_2 = lb

    X = initialization(npop, dim, ub, lb)
    Objective_values = zeros(npop)

    for i = 1:npop
        Objective_values[i] = objfun(X[i, :])
        if Objective_values[i] < Destination_fitness
            Destination_position = X[i, :]
            Destination_fitness = Objective_values[i]
        end
    end

    Convergence_curve[1] = Destination_fitness
    t = 2

    while t <= max_iter
        for i = 1:npop
            for j = 1:dim
                cosh2 = (exp(t / max_iter) + exp(-t / max_iter)) / 2
                sinh2 = (exp(t / max_iter) - exp(-t / max_iter)) / 2
                r1 = rand()
                A = (p - q * (t / max_iter)^(cosh2 / sinh2)) * r1

                if t == BSi
                    ub_2 = Destination_position[j] + (1 - t / max_iter) * abs(Destination_position[j] - Destination_position_second[j])
                    lb_2 = Destination_position[j] - (1 - t / max_iter) * abs(Destination_position[j] - Destination_position_second[j])
                    ub_2 = min(ub_2, ub)
                    lb_2 = max(lb_2, lb)
                    X = initialization(npop, dim, ub_2, lb_2)
                    BSi_temp = BSi
                    BSi = 0
                end

                if t <= T
                    r2, r3, r4, r5 = rand(), rand(), rand(), rand()
                    a1 = 3 * (-1.3 * t / max_iter + m)
                    sinh = (exp(r3) - exp(-r3)) / 2
                    cosh = (exp(r3) + exp(-r3)) / 2
                    W1 = r2 * a1 * (cosh + u * sinh - 1)

                    if A > 1
                        X[i, j] = Destination_position[j] + r4 * W1 * X[i, j] * (r5 <= 0.5 ? 1 : -1)
                    else
                        W3 = r2 * a1 * (cosh + u * sinh)
                        X[i, j] = Destination_position[j] + r4 * W3 * X[i, j] * (r5 <= 0.5 ? 1 : -1)
                    end
                else
                    r2, r3, r4, r5 = rand(), rand(), rand(), rand()
                    a2 = 2 * (-t / max_iter + n)
                    W2 = r2 * a2
                    sinh = (exp(r3) - exp(-r3)) / 2
                    cosh = (exp(r3) + exp(-r3)) / 2

                    if A < 1
                        X[i, j] += r5 * sinh / cosh * abs(W2 * Destination_position[j] - X[i, j])
                    else
                        X[i, j] += (r4 <= 0.5 ? abs(0.003 * W2 * Destination_position[j] - X[i, j]) : -abs(0.003 * W2 * Destination_position[j] - X[i, j]))
                    end
                end
            end
            BSi = BSi_temp
        end

        for i = 1:npop
            X[i, :] = clamp.(X[i, :], lb_2, ub_2)

            Objective_values[i] = objfun(X[i, :])
            if Objective_values[i] < Destination_fitness
                Destination_position = X[i, :]
                Destination_fitness = Objective_values[i]
            end
        end

        if t == BS
            BSi = BS + 1
            BS += floor(Int, (max_iter - BS) / Alpha)
            sorted_idx = sortperm(Objective_values)
            Position_sort = X[sorted_idx, :]
            Destination_position_second = Position_sort[2, :]
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

function SCHO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SCHO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end