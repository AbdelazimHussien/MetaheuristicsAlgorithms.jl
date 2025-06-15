

function levy(d)
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    u = randn(d) * sigma
    v = randn(d)
    step = u ./ abs.(v).^(1 / beta)
    return step
end

"""
# References:

- Xiao, Y., Cui, H., Khurma, R. A., & Castillo, P. A. (2025). Artificial lemming algorithm: a novel bionic meta-heuristic technique for solving real-world engineering optimization problems. Artificial Intelligence Review, 58(3), 84.

"""
function ALA(N, Max_iter, lb, ub, dim, fobj)
    X = initialization(N, dim, ub, lb)
    Position = zeros(dim)
    Score = Inf
    fitness = zeros(N)
    Convergence_curve = zeros(Max_iter)
    vec_flag = [1, -1]

    for i in 1:N
        fitness[i] = fobj(X[i, :])
        if fitness[i] < Score
            Position = copy(X[i, :])
            Score = fitness[i]
        end
    end

    Iter = 1
    while Iter <= Max_iter
        RB = randn(N, dim)
        F = vec_flag[rand(1:2)]
        theta = 2 * atan(1 - Iter / Max_iter)
        Xnew = similar(X)

        for i in 1:N
            E = 2 * log(1 / rand()) * theta
            if E > 1
                if rand() < 0.3
                    r1 = 2 .* rand(dim) .- 1
                    rand_agent = X[rand(1:N), :]
                    Xnew[i, :] = Position .+ F .* RB[i, :] .* (r1 .* (Position .- X[i, :]) .+ (1 .- r1) .* (X[i, :] .- rand_agent))
                else
                    r2 = rand() * (1 + sin(0.5 * Iter))
                    rand_agent = X[rand(1:N), :]
                    Xnew[i, :] = X[i, :] .+ F .* r2 .* (Position .- rand_agent)
                end
            else
                if rand() < 0.5
                    radius = norm(Position .- X[i, :])
                    r3 = rand()
                    spiral = radius * (sin(2 * pi * r3) + cos(2 * pi * r3))
                    Xnew[i, :] = Position .+ F .* X[i, :] .* spiral * rand()
                else
                    G = 2 * sign(rand() - 0.5) * (1 - Iter / Max_iter)
                    Xnew[i, :] = Position .+ F .* G .* levy(dim) .* (Position .- X[i, :])
                end
            end
        end

        for i in 1:N
            Xnew[i, :] = clamp.(Xnew[i, :], lb, ub)
            newPopfit = fobj(Xnew[i, :])
            if newPopfit < fitness[i]
                X[i, :] = Xnew[i, :]
                fitness[i] = newPopfit
            end
            if fitness[i] < Score
                Position = copy(X[i, :])
                Score = fitness[i]
            end
        end

        Convergence_curve[Iter] = Score
        Iter += 1
    end

    return Score, Position, Convergence_curve
end
