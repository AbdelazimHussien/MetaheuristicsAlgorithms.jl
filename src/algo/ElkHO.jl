"""
# References:

- Al-Betar, M.A., Awadallah, M.A., Braik, M.S. et al.
  Elk herd optimizer: a novel nature-inspired metaheuristic algorithm.
  Artif Intell Rev 57, 48 (2024).
  https://doi.org/10.1007/s10462-023-10680-4
"""
function ElkHO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return ElkHO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function ElkHO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    if length(ub) == 1
        ub = ones(dim) * ub
        lb = ones(dim) * lb
    end

    MalesRate = 0.2
    No_of_Males = round(Int, npop * MalesRate)

    Convergence_curve = zeros(max_iter)

    ElkHerd = initialization(npop, dim, ub, lb)

    BestBull = zeros(dim)
    BestBullFitness = Inf

    ElkHerdFitness = [objfun(ElkHerd[i, :]) for i = 1:npop]

    l = 1
    while l <= max_iter
        sorted_indexes = sortperm(ElkHerdFitness)
        sorted_ELKS_fitness = ElkHerdFitness[sorted_indexes]

        NewElkHerd = copy(ElkHerd)
        NewElkHerdFitness = copy(ElkHerdFitness)

        BestBull = ElkHerd[sorted_indexes[1], :]
        BestBullFitness = sorted_ELKS_fitness[1]

        TransposeFitness = [1 / sorted_ELKS_fitness[i] for i = 1:No_of_Males]

        Familes = zeros(Int, npop)
        for i = (No_of_Males+1):npop
            FemaleIndex = sorted_indexes[i]
            randNumber = rand()
            MaleIndex = 0
            sum_fitness = 0.0

            for j = 1:No_of_Males
                sum_fitness += TransposeFitness[j] / sum(TransposeFitness)
                if sum_fitness > randNumber
                    MaleIndex = j
                    break
                end
            end
            Familes[FemaleIndex] = sorted_indexes[MaleIndex]
        end

        for i = 1:npop
            # Male
            if Familes[i] == 0
                h = rand(1:npop)
                for j = 1:dim
                    NewElkHerd[i, j] = ElkHerd[i, j] + rand() * (ElkHerd[h, j] - ElkHerd[i, j])
                    NewElkHerd[i, j] = clamp(NewElkHerd[i, j], lb[j], ub[j])
                end
            else
                h = rand(1:npop)
                MaleIndex = Familes[i]
                hh = randperm(sum(Familes .== MaleIndex))
                h = 1 + floor(Int, (length(hh) - 1) * rand())
                for j = 1:dim
                    rd = -2 + 4 * rand()
                    NewElkHerd[i, j] = ElkHerd[i, j] + (ElkHerd[Familes[i], j] - ElkHerd[i, j]) + rd * (ElkHerd[h, j] - ElkHerd[i, j])
                end
            end
        end

        for i = 1:npop
            NewElkHerdFitness[i] = objfun(NewElkHerd[i, :])
            if NewElkHerdFitness[i] < BestBullFitness
                BestBull = NewElkHerd[i, :]
                BestBullFitness = NewElkHerdFitness[i]
            end
        end

        NewPopulation = vcat(ElkHerd, NewElkHerd)
        NewFitness = [objfun(NewPopulation[i, :]) for i in axes(NewPopulation, 1)]

        sorted_indexes = sortperm(NewFitness)
        sorted_NewFitness = NewFitness[sorted_indexes]

        for i = 1:npop
            ElkHerd[i, :] = NewPopulation[sorted_indexes[i], :]
            ElkHerdFitness[i] = sorted_NewFitness[i]
        end

        Convergence_curve[l] = BestBullFitness

        l += 1
    end

    # return BestBullFitness, BestBull, Convergence_curve
    return OptimizationResult(
        BestBull,
        BestBullFitness,
        Convergence_curve)
end

function ElkHO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return ElkHO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end