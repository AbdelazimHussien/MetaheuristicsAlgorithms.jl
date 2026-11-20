
function Mutation(z, x, b, dim)
    for j = 1:dim
        if rand() < 0.05
            z[j] = x[j]
        end
        if rand() < 0.2
            z[j] = b[j]
        end
    end
    return z
end

function Transborder_reset(z, ub, lb, dim, best)
    for j = 1:dim
        if z[j] > ub[j] || z[j] < lb[j]
            z[j] = best[j]
        end
    end
    return z
end

"""
# References: 

- Yuan, Chong, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, Zongda Wu, and Huiling Chen. "Artemisinin optimization based on malaria therapy: Algorithm and applications to medical image segmentation." Displays 84 (2024): 102740.

"""
function ArtemisininO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return ArtemisininO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function ArtemisininO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb) # (objfun, lb, ub, dim, npop, MaxFEs)
    # Initialization parameters
    FEs = 0
    it = 1
    MaxFEs = npop * max_iter

    # Initialization of the solution set
    pop = initialization(npop, dim, ub, lb)

    # Calculate the fitness value of the initial solution set
    Fitness = zeros(npop)
    for i = 1:npop
        Fitness[i] = objfun(pop[i, :])
        FEs += 1
    end
    fmin, x = findmin(Fitness)

    # Containers
    New_pop = zeros(npop, dim)
    Fitnorm = zeros(npop)
    Convergence_curve = [fmin]

    # Record the current optimal solution
    best = pop[x, :]
    bestfitness = fmin

    # Main loop
    while FEs < MaxFEs
        K = 1 - (FEs^(1 / 6) / MaxFEs^(1 / 6))
        E = exp(-4 * (FEs / MaxFEs))

        for i = 1:npop
            Fitnorm[i] = (Fitness[i] - minimum(Fitness)) / (maximum(Fitness) - minimum(Fitness))

            for j = 1:dim
                if rand() < K
                    if rand() < 0.5
                        New_pop[i, j] = pop[i, j] + E * pop[i, j] * (-1)^FEs
                    else
                        New_pop[i, j] = pop[i, j] + E * best[j] * (-1)^FEs
                    end
                else
                    New_pop[i, j] = pop[i, j]
                end

                if rand() < Fitnorm[i]
                    A = randperm(npop)
                    beta = (rand() / 2) + 0.1
                    New_pop[i, j] = pop[A[3], j] + beta * (pop[A[1], j] - pop[A[2], j])
                end
            end

            # Apply Mutation and Transborder_reset
            New_pop[i, :] = Mutation(New_pop[i, :], pop[i, :], best, dim)
            New_pop[i, :] = Transborder_reset(New_pop[i, :], ub, lb, dim, best)

            # Evaluate new fitness
            tFitness = objfun(New_pop[i, :])
            println("FEs $FEs, tFitness $tFitness")
            FEs += 1

            if tFitness < Fitness[i]
                pop[i, :] = New_pop[i, :]
                Fitness[i] = tFitness
            end
        end

        fmin, x = findmin(Fitness)
        if fmin < bestfitness
            best = pop[x, :]
            bestfitness = fmin
        end

        push!(Convergence_curve, bestfitness)
        it += 1
    end
    println("MaxFEs ", MaxFEs)

    # return bestfitness, best, Convergence_curve
    return OptimizationResult(
        best,
        bestfitness,
        Convergence_curve)
end

function ArtemisininO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return ArtemisininO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end