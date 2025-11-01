"""
# References:

- Ghasemi, M., Golalipour, K., Zare, M., Mirjalili, S., Trojovský, P., Abualigah, L. and Hemmati, R., 2024.
  Flood algorithm (FLA): an efficient inspired meta-heuristic for engineering optimization.
  The Journal of Supercomputing, 80(15), pp.22913-23017.
"""
function FLoodA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return FLoodA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function FLoodA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    CostFunction(x) = objfun(x)

    lb = lb
    ub = ub
    VarSize = (1, dim)
    NFEs = 0
    Ne = 5
    Convergence_curve = []

    Position = zeros(npop, dim)
    println("1 - size(Position) ", size(Position))
    Cost = zeros(npop)
    Val = zeros(npop, dim)

    Position = initialization(npop, dim, lb, ub)
    for i = 1:npop
        Cost[i] = CostFunction(Position[i, :])
    end

    BestSolCost = Inf
    BestSolPosition = nothing

    for i = 1:npop
        if Cost[i] <= BestSolCost
            BestSolCost = Cost[i]
            BestSolPosition = Position[i, :]
        end
    end

    it = 0
    time1 = 0
    while NFEs < npop * max_iter
        it += 1
        PK = ((((max_iter * (it^2) + 1)^0.5 + (1 / ((max_iter / 4) * it)) * log(((max_iter * (it^2) + 1)^0.5 + (max_iter / 4) * it))))^(-2 / 3)) * (1.2 / it)

        for i = 1:npop
            sorted_costs = sort(Cost)
            Pe_i = ((Cost[i] - sorted_costs[1]) / (sorted_costs[end] - sorted_costs[1]))^2
            A = filter(x -> x != i, collect(1:npop))
            a = A[rand(1:end)]

            if rand() > (rand() + Pe_i)
                Val[i, :] = ((PK^randn()) / it) * (rand(dim) .* (ub - lb) .+ lb)
                new_position = Position[i, :] .+ Val[i, :]
            else
                new_position = BestSolPosition .+ rand(dim) .* (Position[a, :] .- Position[i, :])
            end

            new_position = clamp.(new_position, lb, ub)

            new_cost = CostFunction(new_position)
            NFEs += 1

            if new_cost <= Cost[i]
                Position[i, :] = new_position
                Cost[i] = new_cost
            end

            if new_cost <= BestSolCost
                BestSolCost = new_cost
                BestSolPosition = new_position
            end
        end

        Pt = abs(sin(rand() / it))
        if rand() < Pt
            sorted_indices = sortperm(Cost)
            Position = Position[sorted_indices[1:npop-Ne], :]
            Cost = Cost[sorted_indices[1:npop-Ne]]
            Val = Val[sorted_indices[1:npop-Ne], :]

            for _ = 1:Ne
                new_position = BestSolPosition .+ rand() .* (rand(dim) .* (ub - lb) .+ lb)
                new_cost = CostFunction(new_position)
                NFEs += 1

                new_position = vec(new_position)
                Position = vcat(Position, new_position')
                push!(Cost, new_cost)
                Val = vcat(Val, zeros(1, dim))

                if new_cost <= BestSolCost
                    BestSolCost = new_cost
                    BestSolPosition = new_position
                end
            end
        end

        push!(Convergence_curve, BestSolCost)
    end

    Destination_fitness = BestSolCost
    Destination_position = BestSolPosition

    # return Destination_fitness, Destination_position, Convergence_curve
    return OptimizationResult(
        Destination_position,
        Destination_fitness,
        Convergence_curve)
end

function FLoodA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return FLoodA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end