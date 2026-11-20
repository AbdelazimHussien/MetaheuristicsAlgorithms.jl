
# struct AHAResult <: OptimizationResult
#     BestF::Any
#     BestX::Any
#     his_best_fit::Any
# end

"""

    AHA(npop, max_iter, lb, ub, objfun)

The Artificial Hummingbird Algorithm (AHA) is a meta-heuristic optimization algorithm inspired by the foraging behavior of hummingbirds. It is designed to solve numerical optimization problems by simulating the way hummingbirds search for food.

# Arguments:

- `npop`: Number of hummingbirds (population size).
- `max_iter`: Maximum number of iterations.
- `lb`: Lower bounds for the search space.
- `ub`: Upper bounds for the search space.
- `objfun`: Objective function to evaluate the fitness of solutions.

# Returns:
- `AHAResult`: A struct containing:
  - `BestF`: The best fitness value found.
  - `BestX`: The position corresponding to the best fitness.
  - `HisBestFit`: A vector of best fitness values at each iteration.

# References: 

- Zhao, Weiguo, Liying Wang, and Seyedali Mirjalili. "Artificial hummingbird algorithm: A new bio-inspired optimizer with its engineering applications." Computer Methods in Applied Mechanics and Engineering 388 (2022): 114194.
"""
function AHA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return AHA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function AHA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)

    # PopPos = rand(npop, dim) .* (ub .- lb) .+ lb
    PopPos = initialization(npop, dim, ub, lb)
    PopFit = zeros(npop)

    for i = 1:npop
        PopFit[i] = objfun(PopPos[i, :])
    end

    BestF = Inf
    BestX = []

    for i = 1:npop
        if PopFit[i] <= BestF
            BestF = PopFit[i]
            BestX = PopPos[i, :]
        end
    end

    his_best_fit = zeros(max_iter)
    VisitTable = zeros(npop, npop)
    for i = 1:npop
        VisitTable[i, i] = -Inf
    end

    for It = 1:max_iter
        DirectVector = zeros(npop, dim)

        for i = 1:npop
            r = rand()
            if r < 1 / 3
                RandDim = randperm(dim)
                RandNum = dim >= 3 ? ceil(Int, rand() * (dim - 2) + 1) : ceil(Int, rand() * (dim - 1) + 1)
                DirectVector[i, RandDim[1:RandNum]] .= 1
            else
                if r > 2 / 3
                    DirectVector[i, :] .= 1.0
                else
                    RandNum = ceil(Int, rand() * dim)
                    DirectVector[i, RandNum] = 1.0
                end
            end

            if rand() < 0.5
                MaxUnvisitedTime, TargetFoodIndex = findmax(VisitTable[i, :])
                MUT_Index = findall(VisitTable[i, :] .== MaxUnvisitedTime)

                if length(MUT_Index) > 1
                    _, Ind = findmin(PopFit[MUT_Index])
                    TargetFoodIndex = MUT_Index[Ind]
                end

                newPopPos = PopPos[TargetFoodIndex, :] .+ randn() * DirectVector[i, :] .* (PopPos[i, :] .- PopPos[TargetFoodIndex, :])
                newPopPos .= max.(min.(newPopPos, ub), lb)
                newPopFit = objfun(newPopPos)

                if newPopFit < PopFit[i]
                    PopFit[i] = newPopFit
                    PopPos[i, :] = newPopPos
                    VisitTable[i, :] .+= 1
                    VisitTable[i, TargetFoodIndex] = 0
                    VisitTable[:, i] .= maximum(VisitTable, dims=2) .+ 1
                    VisitTable[i, i] = -Inf
                else
                    VisitTable[i, :] .+= 1
                    VisitTable[i, TargetFoodIndex] = 0
                end
            else
                newPopPos = PopPos[i, :] .+ randn() * DirectVector[i, :] .* PopPos[i, :]
                newPopPos .= max.(min.(newPopPos, ub), lb)
                newPopFit = objfun(newPopPos)

                if newPopFit < PopFit[i]
                    PopFit[i] = newPopFit
                    PopPos[i, :] = newPopPos
                    VisitTable[i, :] .+= 1
                    VisitTable[:, i] .= maximum(VisitTable, dims=2) .+ 1
                    VisitTable[i, i] = -Inf
                else
                    VisitTable[i, :] .+= 1
                end
            end
        end

        if It % (2 * npop) == 0
            _, MigrationIndex = findmax(PopFit)
            PopPos[MigrationIndex, :] = rand(dim) .* (ub .- lb) .+ lb
            PopFit[MigrationIndex] = objfun(PopPos[MigrationIndex, :])
            VisitTable[MigrationIndex, :] .+= 1
            VisitTable[:, MigrationIndex] .= maximum(VisitTable, dims=2) .+ 1
            VisitTable[MigrationIndex, MigrationIndex] = -Inf
        end

        for i = 1:npop
            if PopFit[i] < BestF
                BestF = PopFit[i]
                BestX = PopPos[i, :]
            end
        end

        his_best_fit[It] = BestF
    end

    # return  BestF, BestX,his_best_fit
    return OptimizationResult(
        BestX,
        BestF,
        his_best_fit)
end

function AHA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return AHA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end