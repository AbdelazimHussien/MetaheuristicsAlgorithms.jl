
function Mutation(z, x, b, dim)
    for j in 1:dim
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
    for j in 1:dim
        if z[j] > ub || z[j] < lb
            z[j] = best[j]
        end
    end
    return z
end

"""
# References: 

- Yuan, Chong, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, Zongda Wu, and Huiling Chen. "Artemisinin optimization based on malaria therapy: Algorithm and applications to medical image segmentation." Displays 84 (2024): 102740.

"""
function ArtemisininO(N, MaxIter, lb, ub, dim, fobj) # (fobj, lb, ub, dim, N, MaxFEs)
    # Initialization parameters
    FEs = 0
    it = 1
    MaxFEs = N * MaxIter

    # Initialization of the solution set
    pop = initialization(N, dim, ub, lb)
    
    # Calculate the fitness value of the initial solution set
    Fitness = zeros(N)
    for i in 1:N
        Fitness[i] = fobj(pop[i, :])
        FEs += 1
    end
    fmin, x = findmin(Fitness)

    # Containers
    New_pop = zeros(N, dim)
    Fitnorm = zeros(N)
    Convergence_curve = []

    # Record the current optimal solution
    best = pop[x, :]
    bestfitness = fmin

    # Main loop
    while FEs < MaxFEs
        K = 1 - (FEs^(1/6) / MaxFEs^(1/6))
        E = exp(-4 * (FEs / MaxFEs))

        for i in 1:N
            Fitnorm[i] = (Fitness[i] - minimum(Fitness)) / (maximum(Fitness) - minimum(Fitness))

            for j in 1:dim
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
                    A = randperm(N)
                    beta = (rand() / 2) + 0.1
                    New_pop[i, j] = pop[A[3], j] + beta * (pop[A[1], j] - pop[A[2], j])
                end
            end

            # Apply Mutation and Transborder_reset
            New_pop[i, :] = Mutation(New_pop[i, :], pop[i, :], best, dim)
            New_pop[i, :] = Transborder_reset(New_pop[i, :], ub, lb, dim, best)

            # Evaluate new fitness
            tFitness = fobj(New_pop[i, :])
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

    return bestfitness, best, Convergence_curve
end