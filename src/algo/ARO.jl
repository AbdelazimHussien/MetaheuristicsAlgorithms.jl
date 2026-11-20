"""
# References:

- Wang, Liying, Qingjiao Cao, Zhenxing Zhang, Seyedali Mirjalili, and Weiguo Zhao. "Artificial rabbits optimization: A new bio-inspired meta-heuristic algorithm for solving engineering optimization problems." Engineering Applications of Artificial Intelligence 114 (2022): 105082.

"""
function ARO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return ARO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function ARO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    PopPos = initialization(npop, dim, ub, lb)
    # PopPos = [rand(dim) .* (ub - lb) .+ lb for _ = 1:npop]
    PopFit = [objfun(PopPos[i, :]) for i = 1:npop]

    BestF = Inf
    BestX = zeros(dim)

    # Find initial best solution
    for i = 1:npop
        if PopFit[i] <= BestF
            BestF = PopFit[i]
            BestX = PopPos[i, :]
        end
    end

    HisBestF = zeros(max_iter)

    # Main loop
    for It = 1:max_iter
        Direct1 = zeros(npop, dim)
        Direct2 = zeros(npop, dim)
        theta = 2 * (1 - It / max_iter)

        for i = 1:npop
            L = (exp(1) - exp(((It - 1) / max_iter)^2)) * sin(2 * π * rand())
            rd = ceil(Int, rand() * dim)
            Direct1[i, randperm(dim)[1:rd]] .= 1
            c = Direct1[i, :]
            R = L .* c

            A = 2 * log(1 / rand()) * theta

            if A > 1
                K = [1:i-1; i+1:npop]
                RandInd = K[rand(1:end)]
                newPopPos = PopPos[RandInd, :] .+ R .* (PopPos[i, :] .- PopPos[RandInd, :]) .+
                            round(0.5 * (0.05 + rand())) * randn(dim)
            else
                Direct2[i, ceil(Int, rand() * dim)] = 1
                gr = Direct2[i, :]
                H = ((max_iter - It + 1) / max_iter) * randn()
                b = PopPos[i, :] .+ H * gr .* PopPos[i, :]
                newPopPos = PopPos[i, :] .+ R .* (rand() .* b .- PopPos[i, :])
            end

            # Apply boundary constraints
            newPopPos = max.(min.(newPopPos, ub), lb)

            # Evaluate the new solution
            newPopFit = objfun(newPopPos)
            if newPopFit < PopFit[i]
                PopFit[i] = newPopFit
                PopPos[i, :] = newPopPos
            end
        end

        # Update the best solution
        for i = 1:npop
            if PopFit[i] < BestF
                BestF = PopFit[i]
                BestX = PopPos[i, :]
            end
        end

        # Record the best fitness value
        HisBestF[It] = BestF
    end

    # return BestF, BestX, HisBestF
    return OptimizationResult(
        BestX,
        BestF,
        HisBestF)
end

function ARO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return ARO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end