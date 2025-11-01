"""
# References:

- Yang, Yutao, Huiling Chen, Ali Asghar Heidari, and Amir H. Gandomi.
  "Hunger games search: Visions, conception, implementation, deep analysis, perspectives, and towards performance shifts."
  Expert Systems with Applications 177 (2021): 114864.
"""
function HGS(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return HGS(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function HGS(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    bestPositions = zeros(Float64, dim)
    tempPosition = zeros(Float64, npop, dim)
    Destination_fitness = Inf
    Worstest_fitness = -Inf
    AllFitness = fill(Inf, npop)
    VC1 = ones(npop)

    weight3 = ones(Float64, npop, dim)
    weight4 = ones(Float64, npop, dim)

    X = initialization(npop, dim, ub, lb)
    Convergence_curve = zeros(Float64, max_iter)
    it = 1
    hungry = zeros(Float64, npop)
    count = 0

    # Main loop
    while it <= max_iter
        VC2 = 0.03
        sumHungry = 0

        for i = 1:npop
            X[i, :] = clamp.(X[i, :], lb, ub)
            AllFitness[i] = objfun(X[i, :])
        end

        IndexSorted = sortperm(AllFitness) # sortperm(AllFitness, rev=false)  # Get the indices that would sort the array
        AllFitnessSorted = AllFitness[IndexSorted]
        bestFitness = AllFitnessSorted[1]
        worstFitness = AllFitnessSorted[end]

        if bestFitness < Destination_fitness
            bestPositions = X[IndexSorted[1], :]
            Destination_fitness = bestFitness
            count = 0
        end

        if worstFitness > Worstest_fitness
            Worstest_fitness = worstFitness
        end

        for i = 1:npop
            VC1[i] = sech(abs(AllFitness[i] - Destination_fitness))

            if Destination_fitness == AllFitness[i]
                hungry[i] = 0
                count += 1
                tempPosition[count, :] = X[i, :]
            else
                temprand = rand()
                c = (AllFitness[i] - Destination_fitness) / (Worstest_fitness - Destination_fitness) * temprand * 2 * (ub - lb)
                b = ifelse(c < 100, 100 * (1 + temprand), c)
                hungry[i] += maximum(b)
                sumHungry += hungry[i]
            end
        end

        for i = 1:npop
            for j = 2:dim
                weight3[i, j] = (1 - exp(-abs(hungry[i] - sumHungry))) * rand() * 2
                if rand() < VC2
                    weight4[i, j] = hungry[i] * npop / sumHungry * rand()
                else
                    weight4[i, j] = 1
                end
            end
        end

        shrink = 2 * (1 - it / max_iter)
        for i = 1:npop
            if rand() < VC2
                X[i, :] *= 1 + randn()
            else
                A = rand(1:count)
                for j = 1:dim
                    r = rand()
                    vb = 2 * shrink * r - shrink
                    if r > VC1[i]
                        X[i, j] = weight4[i, j] * tempPosition[A, j] + vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
                    else
                        X[i, j] = weight4[i, j] * tempPosition[A, j] - vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
                    end
                end
            end
        end

        Convergence_curve[it] = Destination_fitness
        it += 1
    end

    # return Destination_fitness, bestPositions, Convergence_curve
    return OptimizationResult(
        bestPositions,
        Destination_fitness,
        Convergence_curve)
end

function HGS(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return HGS(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end