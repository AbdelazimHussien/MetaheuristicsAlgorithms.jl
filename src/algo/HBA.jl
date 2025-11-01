"""
# References:

- Hashim, Fatma A., Essam H. Houssein, Kashif Hussain, Mai S. Mabrouk, and Walid Al-Atabany.
  "Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems."
  Mathematics and Computers in Simulation 192 (2022): 84-110.
"""
function HBA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return HBA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function HBA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    beta = 6
    C = 2
    vec_flag = [1.0, -1.0]

    # X = lb .+ rand(npop, dim) .* (ub .- lb)
    X = initialization(npop, dim, ub, lb)
    fitness = ones(npop) * Inf

    for i in 1:npop
        fitness[i] = objfun(vec(X[i,:]'))
    end

    GYbest, gbest = findmin(fitness)
    Xprey = X[gbest, :]
    Xnew = zeros(npop, dim)

    CNVG = zeros(Float64, max_iter)

    for t in 1:max_iter
        alpha = C * exp(-t / max_iter)
        I = Intensity(npop, Xprey, X)

        for i in 1:npop
            r = rand()
            F = vec_flag[Int(floor(2 * rand())) + 1]

            for j in 1:dim
                di = Xprey[j] - X[i, j]

                if r < 0.5
                    r3 = rand()
                    r4 = rand()
                    r5 = rand()


                    Xnew[i, j] = Xprey[j] + F * beta * I[i] * Xprey[j] + F * r3 * alpha * di * abs(cos(2 * π * r4) * (1 - cos(2 * π * r5)))


                else
                    r7 = rand()

                    Xnew[i, j] = Xprey[j] + F * r7 * alpha * di
                end
            end

            Xnew[i, :] = clamp.(Xnew[i, :], lb, ub)


            tempFitness = objfun(vec(Xnew[i,:]'))
            if tempFitness < fitness[i]
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]
            end
        end

        FU = X .> ub
        FL = X .< lb
        X = (X .* .! (FU .| FL)) .+ ub .* FU .+ lb .* FL

        Ybest, index = findmin(fitness)
        CNVG[t] = Ybest
        if Ybest < GYbest
            GYbest = Ybest
            Xprey = X[index, :]
        end
    end


    # return GYbest, Xprey, CNVG
    return OptimizationResult(
        Xprey,
        GYbest,
        CNVG)
end

function HBA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return HBA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function fun_calcobjfunc(func, X)
    npop = size(X, 1)
    Y = zeros(Float64, npop)

    for i in 1:npop
        Y[i] = func(vec(X[i, :]'))
    end

    return Y
end

function Intensity(npop, Xprey, X)
    di = zeros(Float64, npop)
    S = zeros(Float64, npop)
    I = zeros(Float64, npop)
    eps = 1e-10

    for i in 1:npop-1
        di[i] = norm(X[i, :] - Xprey .+ eps)^2
        S[i] = norm(X[i, :] - X[i + 1, :] .+ eps)^2
    end

    di[npop] = norm(X[npop, :] - Xprey .+ eps)^2
    S[npop] = norm(X[npop, :] - X[1, :] .+ eps)^2

    for i in 1:npop
        r2 = rand()
        I[i] = r2 * S[i] / (4 * π * di[i])
    end

    return I
end