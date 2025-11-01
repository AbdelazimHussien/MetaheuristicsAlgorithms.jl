"""
# References:

- Ghasemi, Mojtaba, Mohsen Zare, Pavel Trojovský, Ravipudi Venkata Rao, Eva Trojovská, and Venkatachalam Kandasamy.
  "Optimization based on the smart behavior of plants with its engineering applications: Ivy algorithm."
  Knowledge-Based Systems 295 (2024): 111850.
"""
function IVYA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return IVYA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function IVYA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # Position = rand(npop, dim) .* (ub .- lb) .+ lb
    Position = initialization(npop, dim, ub, lb)
    GV = Position ./ (ub .- lb)
    Cost = [objfun(Position[i, :]) for i = 1:npop]

    BestCosts = zeros(Float64, max_iter)
    Convergence_curve = zeros(Float64, max_iter)

    for it = 1:max_iter
        Costs = Cost
        BestCost = minimum(Costs)
        WorstCost = maximum(Costs)

        newpop = []

        for i = 1:npop
            ii = i + 1 > npop ? 1 : i + 1
            beta_1 = 1 + rand() / 2

            if Cost[i] < beta_1 * Cost[1]
                new_position = Position[i, :] .+ abs.(randn(dim)) .* (Position[ii, :] .- Position[i, :]) .+ randn(dim) .* (GV[i, :])
            else
                new_position = Position[1, :] .* (rand() .+ randn(dim) .* (GV[i, :]))
            end

            GV[i, :] = GV[i, :] .* ((rand()^2) * randn(dim))

            new_position = clamp.(new_position, lb, ub)

            new_GV = new_position ./ (ub .- lb)

            new_cost = objfun(vec(new_position))

            push!(newpop, (new_position, new_cost, new_GV))
        end

        for (new_position, new_cost, new_GV) in newpop
            Position = vcat(Position, new_position')
            push!(Cost, new_cost)
            GV = vcat(GV, new_GV')
        end

        sorted_indices = sortperm(Cost)
        Position = Position[sorted_indices, :]
        Cost = Cost[sorted_indices]

        if length(Cost) > npop
            Position = Position[1:npop, :]
            Cost = Cost[1:npop]
            GV = GV[1:npop, :]
        end

        BestCosts[it] = Cost[1]
        Convergence_curve[it] = Cost[1]
    end

    Destination_fitness = Cost[1]
    Destination_position = Position[1, :]

    # return Destination_fitness, Destination_position, Convergence_curve
    return OptimizationResult(
        Destination_position,
        Destination_fitness,
        Convergence_curve)
end

function IVYA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return IVYA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end