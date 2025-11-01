function noh(best_hyena_fitness)
    min_val = 0.5
    max_val = 1.0
    count = 0

    M = (max_val - min_val) * rand() + min_val
    M += best_hyena_fitness[1]

    for i = 2:lastindex(best_hyena_fitness)
        if M >= best_hyena_fitness[i]
            count += 1
        end
    end

    return count
end

"""
# References:

- Dhiman, Gaurav, and Vijay Kumar.
  "Spotted hyena optimizer: a novel bio-inspired based metaheuristic technique for engineering applications."
  Advances in Engineering Software 114 (2017): 48-70.
"""
function SHO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SHO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SHO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    hyena_pos = initialization(npop, dim, ub, lb)
    convergence_curve = zeros(max_iter)
    Iteration = 1

    pre_population = zeros(npop, dim)
    best_hyenas = zeros(npop, dim)
    best_hyena_pos = zeros(npop)
    best_hyena_score = Inf

    while Iteration < max_iter
        hyena_fitness = zeros(npop)

        pre_fitness = zeros(npop)
        best_hyena_fitness = 0

        for i in axes(hyena_pos, 1)
            H_ub = hyena_pos[i, :] .> ub
            H_lb = hyena_pos[i, :] .< lb
            hyena_pos[i, :] = (hyena_pos[i, :] .* .~(H_ub .+ H_lb)) .+ ub .* H_ub .+ lb .* H_lb
            hyena_fitness[i] = objfun(hyena_pos[i, :])
        end

        if Iteration == 1
            FS = sortperm(hyena_fitness)
            fitness_sorted = hyena_fitness[FS]
            sorted_population = hyena_pos[FS, :]
            best_hyenas = sorted_population
            best_hyena_fitness = fitness_sorted
        else
            double_population = vcat(pre_population, best_hyenas)
            double_fitness = vcat(pre_fitness, best_hyena_fitness)
            FS = sortperm(double_fitness)
            double_fitness_sorted = sort(double_fitness)
            double_sorted_population = double_population[FS, :]
            fitness_sorted = double_fitness_sorted[1:npop]
            sorted_population = double_sorted_population[1:npop, :]
            best_hyenas = sorted_population
            best_hyena_fitness = fitness_sorted
        end

        NOH = noh(best_hyena_fitness)

        best_hyena_score = fitness_sorted[1]
        best_hyena_pos = sorted_population[1, :]
        pre_population = copy(hyena_pos)
        pre_fitness = copy(hyena_fitness)

        a = 5 - Iteration * (5 / max_iter)
        CV = 0

        for i in axes(hyena_pos, 1)
            for i in axes(hyena_pos, 2)
                CV = 0
                for k = 1:NOH
                    r1 = rand()
                    r2 = rand()
                    Var1 = 2 * a * r1 - a
                    Var2 = 2 * r2
                    distance_to_hyena = abs(Var2 * sorted_population[k, j] - hyena_pos[i, j])
                    HYE = sorted_population[k, j] - Var1 * distance_to_hyena
                    CV += HYE
                end
                hyena_pos[i, j] = CV / (NOH + 1)
            end
        end

        convergence_curve[Iteration] = best_hyena_score
        Iteration += 1
    end

    # return best_hyena_score, best_hyena_pos, convergence_curve
    return OptimizationResult(
        best_hyena_pos,
        best_hyena_score,
        convergence_curve)
end

function SHO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SHO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end