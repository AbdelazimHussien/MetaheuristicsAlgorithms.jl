"""
# References:

- Zhong, C., Li, G., Meng, Z., Li, H., Yildiz, A. R., & Mirjalili, S. (2025).
  Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers.
  Neural Computing and Applications, 37(5), 3641-3683.
"""
function SFOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SFOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SFOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # Starfish Optimization Algorithm (SFOA)
    GP = 0.5  # parameter

    # Ensure bounds are vectors of length nD
    lb = length(lb) == 1 ? fill(lb, dim) : lb
    ub = length(ub) == 1 ? fill(ub, dim) : ub

    fvalbest = Inf
    Curve = zeros(max_iter)
    # Xpos = [lb .+ rand(dim) .* (ub .- lb) for _ = 1:npop]
    Xpos = initialization(npop, dim, ub, lb)
    Fitness = [objfun(X) for X in Xpos]

    fvalbest, order = findmin(Fitness)
    xposbest = copy(Xpos[order])

    newX = [zeros(dim) for _ = 1:npop]

    T = 1
    while T <= max_iter
        theta = π / 2 * T / max_iter
        tEO = (max_iter - T) / max_iter * cos(theta)

        if rand() < GP  # exploration
            for i = 1:npop
                if dim > 5
                    jp1 = randperm(dim)[1:5]
                    for j in jp1
                        pm = (2rand() - 1) * π
                        if rand() < GP
                            newX[i][j] = Xpos[i][j] + pm * (xposbest[j] - Xpos[i][j]) * cos(theta)
                        else
                            newX[i][j] = Xpos[i][j] - pm * (xposbest[j] - Xpos[i][j]) * sin(theta)
                        end
                        if newX[i][j] > ub[j] || newX[i][j] < lb[j]
                            newX[i][j] = Xpos[i][j]
                        end
                    end
                else
                    jp2 = rand(1:dim)
                    im = randperm(npop)
                    rand1 = 2rand() - 1
                    rand2 = 2rand() - 1
                    newX[i][jp2] = tEO * Xpos[i][jp2] + rand1 * (Xpos[im[1]][jp2] - Xpos[i][jp2]) + rand2 * (Xpos[im[2]][jp2] - Xpos[i][jp2])
                    if newX[i][jp2] > ub[jp2] || newX[i][jp2] < lb[jp2]
                        newX[i][jp2] = Xpos[i][jp2]
                    end
                end
                newX[i] = clamp.(newX[i], lb, ub)
            end
        else  # exploitation
            df = randperm(npop)[1:5]
            dm = [xposbest .- Xpos[d] for d in df]
            for i = 1:npop
                r1, r2 = rand(), rand()
                kp = randperm(5)[1:2]
                newX[i] = Xpos[i] .+ r1 .* dm[kp[1]] .+ r2 .* dm[kp[2]]
                if i == npop
                    newX[i] = exp(-T * npop / max_iter) .* Xpos[i]
                end
                newX[i] = clamp.(newX[i], lb, ub)
            end
        end

        # Fitness evaluation
        for i = 1:npop
            newFit = objfun(newX[i])
            if newFit < Fitness[i]
                Fitness[i] = newFit
                Xpos[i] = copy(newX[i])
                if newFit < fvalbest
                    fvalbest = newFit
                    xposbest = copy(Xpos[i])
                end
            end
        end

        Curve[T] = fvalbest
        T += 1
    end

    #  return TargetFitness, TargetPosition, Convergence_curve,
    # return fvalbest, xposbest, Curve
    return OptimizationResult(
        best_hyena_pos,
        best_hyena_score,
        convergence_curve)
end

function SFOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SFOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end