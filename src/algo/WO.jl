# function levyFlight(d::Int)
#     beta = 3/2

#     sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1/beta)
#     u = randn(d) * sigma
#     v = randn(d)
#     step = u ./ abs.(v).^(1/beta)

#     return step
# end

"""
# References:

- Han, Muxuan, Zunfeng Du, Kum Fai Yuen, Haitao Zhu, Yancang Li, and Qiuyu Yuan.
  "Wave optimization algorithm: A new metaheuristic algorithm for solving optimization problems."
  Knowledge-Based Systems 236 (2022): 107760.
"""
function WO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return WO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function WO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    best_pos = zeros(Float64, dim)
    Second_Pos = zeros(Float64, dim)
    best_score = Inf
    Second_Score = Inf
    GBestX = repeat(best_pos, npop, 1)


    X = initialization(npop, dim, ub, lb)

    convergence_curve = zeros(Float64, max_iter)

    P = 0.4
    F_number = round(Int, npop * P)
    M_number = F_number
    C_number = npop - F_number - M_number

    t = 0

    while t < max_iter
        for i in axes(X, 1)
            Flag4ub = X[i, :] .> ub
            Flag4lb = X[i, :] .< lb
            X[i, :] .= (X[i, :] .* .! (Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness = objfun(X[i, :])

            if fitness < best_score
                best_score = fitness
                best_pos .= X[i, :]
            end

            if fitness > best_score && fitness < Second_Score
                Second_Score = fitness
                Second_Pos .= X[i, :]
            end
        end

        Alpha = 1 - t / max_iter
        Beta = 1 - 1 / (1 + exp((1 / 2 * max_iter - t) / max_iter * 10))
        A = 2 * Alpha
        r1 = rand()
        R = 2 * r1 - 1
        Danger_signal = A * R
        r2 = rand()
        Satey_signal = r2

        if abs(Danger_signal) >= 1
            r3 = rand()
            Rs = size(X, 1)
            Migration_step = (Beta * r3^2) * (X[randperm(Rs), :] .- X[randperm(Rs), :])
            X .= X .+ Migration_step

        elseif abs(Danger_signal) < 1
            if Satey_signal >= 0.5
                for i in 1:M_number
                    xy = zeros(Float64, M_number, 1)
                    base = 7
                    xy[i, 1] = hal(i, base)
                    m1 = lb .+ xy[i, :] .* (ub .- lb)
                    X[i, :] .= m1
                end
                for j in M_number + 1:M_number + F_number
                    X[j, :] .= X[j, :] .+ Alpha .* (X[i, :] .- X[j, :]) .+ (1 .- Alpha) .* (GBestX[j, :] .- X[j, :])
                end
                for i in npop - C_number + 1:npop
                    P = rand()
                    o = GBestX[i, :] .+ X[i, :] .* levy(dim)
                    X[i, :] .= P * (o .- X[i, :])
                end
            end

            if Satey_signal < 0.5 && abs(Danger_signal) >= 0.5
                for i in 1:npop
                    r4 = rand()
                    X[i, :] .= X[i, :] .* R .- abs.(GBestX[i, :] .- X[i, :]) .* r4^2
                end
            end

            if Satey_signal < 0.5 && abs(Danger_signal) < 0.5
                for i in axes(X, 1)
                    for j in axes(X, 2)
                        theta1 = rand()
                        a1 = Beta * rand() - Beta
                        b1 = tan(theta1 * π)
                        X1 = best_pos[j] - a1 * b1 * abs(best_pos[j] - X[i, j])

                        theta2 = rand()
                        a2 = Beta * rand() - Beta
                        b2 = tan(theta2 * π)
                        X2 = Second_Pos[j] - a2 * b2 * abs(Second_Pos[j] - X[i, j])

                        X[i, j] = (X1 + X2) / 2
                    end
                end
            end
        end
        t += 1
        convergence_curve[t] = best_score
    end

    # return best_score, best_pos, convergence_curve
    return OptimizationResult(
        best_pos,
        best_score,
        convergence_curve)
end

function WO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)
    return WO(npop, max_iter, problem.lb, problem.ub, problem.dim, problem.objfun)
end
function hal(index, base)
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0
        result += f * mod(i, base)
        i = div(i, base)
        f /= base
    end
    return result
end