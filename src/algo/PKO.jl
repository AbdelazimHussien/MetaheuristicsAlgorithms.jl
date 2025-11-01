"""
# References:

- Bouaouda, Anas, Fatma A. Hashim, Yassine Sayouti, and Abdelazim G. Hussien.
  "Pied kingfisher optimizer: a new bio-inspired algorithm for solving numerical optimization and industrial engineering problems."
  Neural Computing and Applications (2024): 1-59.
"""
function PKO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return PKO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function PKO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    BF = 8
    Crest_angles = 2 * π * rand()

    X = initialization(npop, dim, ub, lb)

    Fitness = zeros(npop)
    Convergence_curve = zeros(max_iter, 1)

    for i = 1:npop
        Fitness[i] = objfun(X[i, :])
    end

    sorted_indexes = sortperm(Fitness)
    Best_position = X[sorted_indexes[1], :]
    Best_fitness = Fitness[sorted_indexes[1]]
    Convergence_curve[1] = Best_fitness

    t = 1
    PEmax = 0.5
    PEmin = 0

    while t < max_iter + 1
        o = exp(-t / max_iter)^2

        for i = 1:npop
            if rand() < 0.8
                j = i
                while i == j
                    seed = randperm(npop)
                    j = seed[1]
                end

                beatingRate = rand() * (Fitness[j]) / Fitness[i]
                alpha = 2 * randn(dim) .- 1

                if rand() < 0.5
                    T = beatingRate - (t^(1 / BF) / max_iter^(1 / BF))
                    X_1 = X[i, :] + alpha .* T .* (X[j, :] - X[i, :])
                else
                    T = (exp(1) - exp(((t - 1) / max_iter)^(1 / BF))) * cos(Crest_angles)
                    X_1 = X[i, :] + alpha .* T .* (X[j, :] - X[i, :])
                end
            else
                alpha = 2 * randn(dim) .- 1
                b = X[i, :] + o^2 * randn() .* Best_position
                HuntingAbility = rand() * (Fitness[i]) / Best_fitness
                X_1 = X[i, :] + HuntingAbility * o * alpha .* (b - Best_position)
            end

            FU = X_1 .> ub
            FL = X_1 .< lb
            X_1 = (X_1 .* .~(FU .+ FL)) + ub .* FU + lb .* FL

            fitnessn = objfun(X_1)

            if fitnessn < Fitness[i]
                Fitness[i] = fitnessn
                X[i, :] = X_1
            end

            if Fitness[i] < Best_fitness
                Best_fitness = Fitness[i]
                Best_position = X[i, :]
            end
        end

        PE = PEmax - (PEmax - PEmin) * (t / max_iter)

        for i = 1:npop
            alpha = 2 * randn(dim) .- 1
            if rand() > (1 - PE)
                X_1 = X[rand(1:npop), :] + o * alpha .* abs.(X[i, :] - X[rand(1:npop), :])
            else
                X_1 = X[i, :]
            end

            FU = X_1 .> ub
            FL = X_1 .< lb
            X_1 = (X_1 .* .~(FU .+ FL)) + ub .* FU + lb .* FL

            fitnessn = objfun(X_1)

            if fitnessn < Fitness[i]
                Fitness[i] = fitnessn
                X[i, :] = X_1
            end

            if Fitness[i] < Best_fitness
                Best_fitness = Fitness[i]
                Best_position = X[i, :]
            end
        end

        Convergence_curve[t] = Best_fitness
        t += 1
    end

    # return Best_fitness, Best_position, Convergence_curve
    return OptimizationResult(
        Best_position,
        Best_fitness,
        Convergence_curve)
end

function PKO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return PKO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end