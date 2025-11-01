"""
# References:

- Zhao, Weiguo, Zhenxing Zhang, and Liying Wang.
  "Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications."
  Engineering Applications of Artificial Intelligence 87 (2020): 103300.
"""
function MRFO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return MRFO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function MRFO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # PopPos = [rand(dim) .* (ub - lb) .+ lb for _ = 1:npop]
    PopPos = initialization(npop, dim, lb, ub)
    PopFit = [objfun(PopPos[i]) for i = 1:npop]

    BestF = Inf
    BestX = []

    for i = 1:npop
        if PopFit[i] <= BestF
            BestF = PopFit[i]
            BestX = PopPos[i]
        end
    end

    his_best_fit = zeros(max_iter)

    for It = 1:max_iter
        Coef = It / max_iter

        if rand() < 0.5
            r1 = rand()
            Beta = 2 * exp(r1 * ((max_iter - It + 1) / max_iter)) * sin(2 * pi * r1)
            if Coef > rand()
                newPopPos1 = BestX .+ rand(dim) .* (BestX .- PopPos[1]) .+ Beta .* (BestX .- PopPos[1])  # Equation (4)
            else
                IndivRand = rand(dim) .* (ub - lb) .+ lb
                newPopPos1 = IndivRand .+ rand(dim) .* (IndivRand .- PopPos[1]) .+ Beta .* (IndivRand .- PopPos[1])  # Equation (7)
            end
        else
            Alpha = 2 * rand(dim) .* (-log.(rand(dim))) .^ 0.5
            newPopPos1 = PopPos[1] .+ rand(dim) .* (BestX .- PopPos[1]) .+ Alpha .* (BestX .- PopPos[1])  # Equation (1)
        end

        newPopPos = [zeros(dim) for _ = 1:npop]
        newPopPos[1] = newPopPos1

        for i = 2:npop
            if rand() < 0.5
                r1 = rand()
                Beta = 2 * exp(r1 * ((max_iter - It + 1) / max_iter)) * sin(2 * pi * r1)
                if Coef > rand()
                    newPopPos[i] = BestX .+ rand(dim) .* (PopPos[i-1] .- PopPos[i]) .+ Beta .* (BestX .- PopPos[i])  # Equation (4)
                else
                    IndivRand = rand(dim) .* (ub - lb) .+ lb
                    newPopPos[i] = IndivRand .+ rand(dim) .* (PopPos[i-1] .- PopPos[i]) .+ Beta .* (IndivRand .- PopPos[i])  # Equation (7)
                end
            else
                Alpha = 2 * rand(dim) .* (-log.(rand(dim))) .^ 0.5
                newPopPos[i] = PopPos[i] .+ rand(dim) .* (PopPos[i-1] .- PopPos[i]) .+ Alpha .* (BestX .- PopPos[i])  # Equation (1)
            end
        end

        newPopFit = zeros(npop)
        for i = 1:npop
            newPopPos[i] = max.(min.(newPopPos[i], ub), lb)
            newPopFit[i] = objfun(newPopPos[i])
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i] = newPopPos[i]
            end
        end

        S = 2
        for i = 1:npop
            newPopPos[i] = PopPos[i] .+ S * (rand() * BestX .- rand() * PopPos[i])  # Equation (8)
        end

        for i = 1:npop
            newPopPos[i] = max.(min.(newPopPos[i], ub), lb)
            newPopFit[i] = objfun(newPopPos[i])
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i] = newPopPos[i]
            end
        end

        for i = 1:npop
            if PopFit[i] < BestF
                BestF = PopFit[i]
                BestX = PopPos[i]
            end
        end

        his_best_fit[It] = BestF
    end
    # return BestF, BestX, his_best_fit
    return OptimizationResult(
        BestX,
        BestF,
        his_best_fit)
end

function MRFO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return MRFO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end