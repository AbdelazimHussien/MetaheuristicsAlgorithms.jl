"""
# References:

- Jia, Heming, et al.
  "Superb Fairy-wren Optimization Algorithm: a novel metaheuristic algorithm for solving feature selection problems."
  Cluster Computing 28.4 (2025): 246.
"""
function SuperbFOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SuperbFOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SuperbFOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    MaxFEs = max_iter * npop
    curve = zeros(MaxFEs)
    X = initialization(npop, dim, ub, lb)
    Xnew = zeros(npop, dim)
    best_fitness = Inf
    best_position = zeros(dim)
    fitness = zeros(npop)
    FEs = 1
    lb = fill(lb, dim)
    ub = fill(ub, dim)

    for i = 1:npop
        fitness[i] = objfun(X[i, :])
        if fitness[i] < best_fitness
            best_fitness = fitness[i]
            best_position = copy(X[i, :])
        end
        FEs += 1
        if FEs <= MaxFEs
            curve[FEs] = best_fitness
        end
    end

    while FEs <= MaxFEs
        C = 0.8
        r1 = rand()
        r2 = rand()
        w = (π / 2) * (FEs / MaxFEs)
        k = 0.2 * sin(π / 2 - w)
        l = 0.5 * levy(npop, dim, 1.5)
        y = rand(1:npop)
        c1 = rand()
        T = 0.5
        m = FEs / MaxFEs * 2
        p = sin.(ub .- lb) .* 2 .+ (ub .- lb) .* m
        Xb = best_position
        XG = best_position .* C

        for i = 1:npop
            if T < c1
                Xnew[i, :] = X[i, :] .+ lb .+ (ub .- lb) .* rand(1, dim)
            else
                s = r1 * 20 + r2 * 20
                if s > 20
                    Xnew[i, :] = Xb .+ X[i, :] .* l[y, :] .* k
                else
                    Xnew[i, :] = XG .+ (Xb .- X[i, :]) .* p
                end
            end
        end

        X .= Xnew

        for i = 1:npop
            Xnew[i, :] = clamp.(Xnew[i, :], lb, ub)
            fitness[i] = objfun(Xnew[i, :])
            if fitness[i] < best_fitness
                best_fitness = fitness[i]
                best_position = copy(Xnew[i, :])
            end
            FEs += 1
            if FEs <= MaxFEs
                curve[FEs] = best_fitness
            else
                break
            end
        end

        if FEs >= MaxFEs
            break
        end
    end

    # return best_fitness, best_position, curve
    return OptimizationResult(
        best_position,
        best_fitness,
        curve)
end

function SuperbFOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SuperbFOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end