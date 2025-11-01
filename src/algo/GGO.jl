"""
# References:

- El-Kenawy, El-Sayed M., Nima Khodadadi, Seyedali Mirjalili, Abdelaziz A. Abdelhamid, Marwa M. Eid, and Abdelhameed Ibrahim.
  "Greylag goose optimization: nature-inspired optimization algorithm."
  Expert Systems with Applications 238 (2024): 122147.
"""
function GGO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GGO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GGO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # population = lb .+ rand(npop, dim) .* (ub .- lb)
    population = initialization(npop, dim, ub, lb)
    fitness = zeros(Float64, npop)
    Convergence = zeros(max_iter, 1)

    for i = 1:npop
        fitness[i] = objfun(population[i, :])
    end

    for iter = 1:max_iter
        for i = 1:npop
            best_agent_index = argmin(fitness)

            new_solution = population[i, :] .+ rand(dim) .* (population[best_agent_index, :] .- population[i, :])

            new_solution = max.(min.(new_solution, ub), lb)

            new_fitness = objfun(new_solution)

            if new_fitness < fitness[i]
                population[i, :] .= new_solution
                fitness[i] = new_fitness
            end
        end
        Convergence[iter], _ = findmin(fitness)
        # println(Convergence[iter])
    end

    best_fitness, best_index = findmin(fitness)
    best_solution = population[best_index, :]

    # return best_fitness, best_solution, Convergence
    return OptimizationResult(
        best_solution,
        best_fitness,
        Convergence)
end

function GGO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GGO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end