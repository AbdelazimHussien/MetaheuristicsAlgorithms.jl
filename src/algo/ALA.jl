
"""
# References:

- Xiao, Y., Cui, H., Khurma, R. A., & Castillo, P. A. (2025). Artificial lemming algorithm: a novel bionic meta-heuristic technique for solving real-world engineering optimization problems. Artificial Intelligence Review, 58(3), 84.
"""
function ALA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return ALA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function ALA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    X = initialization(npop, dim, ub, lb)
    position = zeros(dim)
    score = Inf
    fitness = zeros(npop)
    convergence_curve = zeros(max_iter)
    vec_flag = [1, -1]

    for i = 1:npop
        fitness[i] = objfun(X[i, :])
        if fitness[i] < score
            position = copy(X[i, :])
            score = fitness[i]
        end
    end

    Iter = 1
    while Iter <= max_iter
        RB = randn(npop, dim)
        F = vec_flag[rand(1:2)]
        theta = 2 * atan(1 - Iter / max_iter)
        Xnew = similar(X)

        for i = 1:npop
            E = 2 * log(1 / rand()) * theta
            if E > 1
                if rand() < 0.3
                    r1 = 2 .* rand(dim) .- 1
                    rand_agent = X[rand(1:npop), :]
                    Xnew[i, :] = position .+ F .* RB[i, :] .* (r1 .* (position .- X[i, :]) .+ (1 .- r1) .* (X[i, :] .- rand_agent))
                else
                    r2 = rand() * (1 + sin(0.5 * Iter))
                    rand_agent = X[rand(1:npop), :]
                    Xnew[i, :] = X[i, :] .+ F .* r2 .* (position .- rand_agent)
                end
            else
                if rand() < 0.5
                    radius = norm(position .- X[i, :])
                    r3 = rand()
                    spiral = radius * (sin(2 * pi * r3) + cos(2 * pi * r3))
                    Xnew[i, :] = position .+ F .* X[i, :] .* spiral * rand()
                else
                    G = 2 * sign(rand() - 0.5) * (1 - Iter / max_iter)
                    Xnew[i, :] = position .+ F .* G .* levy(dim) .* (position .- X[i, :])
                end
            end
        end

        for i = 1:npop
            Xnew[i, :] = clamp.(Xnew[i, :], lb, ub)
            newPopfit = objfun(Xnew[i, :])
            if newPopfit < fitness[i]
                X[i, :] = Xnew[i, :]
                fitness[i] = newPopfit
            end
            if fitness[i] < score
                position = copy(X[i, :])
                score = fitness[i]
            end
        end

        convergence_curve[Iter] = score
        Iter += 1
    end

    # return score, position, convergence_curve
    return OptimizationResult(
        position,
        score,
        convergence_curve)
end

function ALA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return ALA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end