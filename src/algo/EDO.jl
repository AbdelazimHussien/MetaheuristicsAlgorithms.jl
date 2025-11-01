"""
# References:

- Abdel-Basset, Mohamed, Doaa El-Shahat, Mohammed Jameel, and Mohamed Abouhawwash.
  "Exponential distribution optimizer (EDO): a novel math-inspired algorithm for global optimization and engineering problems."
  Artificial Intelligence Review 56, no. 9 (2023): 9329-9400.
"""
function EDO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return EDO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function EDO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    BestSol = zeros(dim)
    BestFitness = Inf
    Xwinners = initialization(npop, dim, ub, lb)
    Fitness = zeros(npop)

    for i = 1:npop
        Fitness[i] = objfun(Xwinners[i, :])
        if Fitness[i] < BestFitness
            BestFitness = Fitness[i]
            BestSol = Xwinners[i, :]
        end
    end

    Memoryless = copy(Xwinners)
    iter = 1
    cgcurve = zeros(max_iter)

    while iter <= max_iter
        V = zeros(npop, dim)
        cgcurve[iter] = BestFitness

        sorted_indices = sortperm(Fitness)
        Fitness = Fitness[sorted_indices]
        Xwinners = Xwinners[sorted_indices, :]

        d = (1 - iter / max_iter)
        f = 2 * rand() - 1
        a = f^10
        b = f^5
        c = d * f
        X_guide = mean(Xwinners[1:3, :], dims=1) |> vec

        for i = 1:npop
            alpha = rand()
            if alpha < 0.5
                if Memoryless[i, :] == Xwinners[i, :]
                    Mu = (X_guide .+ Memoryless[i, :]) ./ 2.0
                    ExP_rate = 1.0 ./ Mu
                    variance = 1.0 ./ (ExP_rate .^ 2)
                    V[i, :] = a .* (Memoryless[i, :] .- variance) .+ b .* X_guide
                else
                    Mu = (X_guide .+ Memoryless[i, :]) ./ 2.0
                    ExP_rate = 1.0 ./ Mu
                    variance = 1.0 ./ (ExP_rate .^ 2)
                    phi = rand()
                    V[i, :] = b .* (Memoryless[i, :] .- variance) .+ log(phi) .* Xwinners[i, :]
                end
            else
                M = mean(Xwinners, dims=1) |> vec
                s = randperm(npop)
                D1 = M .- Xwinners[s[1], :]
                D2 = M .- Xwinners[s[2], :]
                Z1 = M .- D1 .+ D2
                Z2 = M .- D2 .+ D1
                V[i, :] = Xwinners[i, :] .+ c .* Z1 .+ (1.0 - c) .* Z2 .- M
            end
            V[i, :] = clamp.(V[i, :], lb, ub)
        end

        Memoryless .= V

        V_Fitness = zeros(npop)
        for i = 1:npop
            V_Fitness[i] = objfun(V[i, :])
            if V_Fitness[i] < Fitness[i]
                Xwinners[i, :] = V[i, :]
                Fitness[i] = V_Fitness[i]
                if Fitness[i] < BestFitness
                    BestFitness = Fitness[i]
                    BestSol = Xwinners[i, :]
                end
            end
        end

        iter += 1
    end

    # return BestFitness, BestSol, cgcurve
    return OptimizationResult(
        BestFitness,
        BestSol,
        cgcurve)
end

function EDO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return EDO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end