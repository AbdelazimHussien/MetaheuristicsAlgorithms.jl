"""
# References:

- Su, Hang, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, and Huiling Chen.
  "RIME: A physics-based optimization."
  Neurocomputing 532 (2023): 183-214.
"""
function RIME(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return RIME(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function RIME(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    println("RIME is now tackling your problem")

    Best_rime = zeros(dim)
    Best_rime_rate = Inf
    Rimepop = initialization(npop, dim, ub, lb)
    lb = fill(lb, dim)
    ub = fill(ub, dim)
    it = 1
    Convergence_curve = zeros(max_iter)
    Rime_rates = zeros(npop)
    newRime_rates = zeros(npop)
    W = 5

    for i = 1:npop
        Rime_rates[i] = objfun(Rimepop[i, :])
        if Rime_rates[i] < Best_rime_rate
            Best_rime_rate = Rime_rates[i]
            Best_rime = Rimepop[i, :]
        end
    end

    while it <= max_iter
        RimeFactor = (rand() - 0.5) * 2 * cos(pi * it / (max_iter / 10)) * (1 - round(it * W / max_iter) / W)  # Eq.(3),(4),(5)
        E = sqrt(it / max_iter)
        newRimepop = copy(Rimepop)
        normalized_rime_rates = normalize(Rime_rates)

        for i = 1:npop
            for j = 1:dim
                r1 = rand()
                if r1 < E
                    newRimepop[i, j] = Best_rime[j] + RimeFactor * ((ub[j] - lb[j]) * rand() + lb[j])  # Eq.(3)
                end
                r2 = rand()
                if r2 < normalized_rime_rates[i]
                    newRimepop[i, j] = Best_rime[j]
                end
            end
        end

        for i = 1:npop
            newRimepop[i, :] = clamp.(newRimepop[i, :], lb, ub)
            newRime_rates[i] = objfun(newRimepop[i, :])

            if newRime_rates[i] < Rime_rates[i]
                Rime_rates[i] = newRime_rates[i]
                Rimepop[i, :] = newRimepop[i, :]
                if newRime_rates[i] < Best_rime_rate
                    Best_rime_rate = Rime_rates[i]
                    Best_rime = Rimepop[i, :]
                end
            end
        end

        Convergence_curve[it] = Best_rime_rate
        println("Convergence_curve[$it] = $Best_rime_rate")
        it += 1
    end

    # return Best_rime_rate, Best_rime, Convergence_curve
    return OptimizationResult(
        Best_rime,
        Best_rime_rate,
        Convergence_curve)
end
function RIME(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return RIME(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end