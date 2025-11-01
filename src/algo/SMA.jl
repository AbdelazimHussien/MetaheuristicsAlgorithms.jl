"""
# References:

- Li, Shimin, Huiling Chen, Mingjing Wang, Ali Asghar Heidari, and Seyedali Mirjalili.
  "Slime mould algorithm: A new method for stochastic optimization."
  Future generation computer systems 111 (2020): 300-323.
"""
function SMA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SMA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SMA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    best_positions = zeros(1, dim)
    destination_fitness = Inf
    AllFitness = fill(Inf, npop)
    weight = ones(npop, dim)
    X = initialization(npop, dim, ub, lb)
    convergence_curve = zeros(max_iter)
    it = 1
    lb = ones(dim) .* lb
    ub = ones(dim) .* ub
    z = 0.03

    while it <= max_iter
        for i = 1:npop
            Flag4ub = X[i, :] .> ub
            Flag4lb = X[i, :] .< lb
            X[i, :] = (X[i, :] .* .~(Flag4ub .+ Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb
            AllFitness[i] = objfun(X[i, :])
        end

        SmellIndex = sortperm(AllFitness)
        SmellOrder = AllFitness[SmellIndex]
        worstFitness = SmellOrder[npop]
        bestFitness = SmellOrder[1]

        S = bestFitness - worstFitness + eps()

        for i = 1:npop
            for j = 1:dim
                if i <= (npop / 2)
                    weight[SmellIndex[i], j] = 1 + rand() * log10((bestFitness - AllFitness[SmellIndex[i]]) / S + 1)
                else
                    weight[SmellIndex[i], j] = 1 - rand() * log10((bestFitness - AllFitness[SmellIndex[i]]) / S + 1)
                end
            end
        end

        if bestFitness < destination_fitness
            best_positions = X[SmellIndex[1], :]
            destination_fitness = bestFitness
        end

        a = atanh(-(it / max_iter) + 1)
        b = 1 - it / max_iter

        for i = 1:npop
            if rand() < z
                X[i, :] = (ub - lb) .* rand(dim) .+ lb
            else
                p = tanh(abs(AllFitness[i] - destination_fitness))
                vb = rand(dim) .* (2a) .- a
                vc = rand(dim) .* (2b) .- b
                for j = 1:dim
                    r = rand()
                    A = rand(1:npop)[1]
                    B = rand(1:npop)[1]
                    if r < p
                        X[i, j] = best_positions[j] + vb[j] * (weight[i, j] * X[A, j] - X[B, j])
                    else
                        X[i, j] = vc[j] * X[i, j]
                    end
                end
            end
        end

        convergence_curve[it] = destination_fitness
        it += 1
    end

    # return destination_fitness, best_positions, convergence_curve
    return OptimizationResult(
        best_positions,
        destination_fitness,
        convergence_curve)
end

function SMA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SMA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end