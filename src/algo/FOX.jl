"""
# References:

- Mohammed, Hardi, and Tarik Rashid.
  "FOX: a FOX-inspired optimization algorithm."
  Applied Intelligence 53, no. 1 (2023): 1030-1050.
"""
function FOX(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return FOX(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function FOX(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Best_pos = zeros(dim)
    Best_score = Inf
    MinT = Inf

    X = initialization(npop, dim, ub, lb)
    Distance_Fox_Rat = zeros(npop, dim)
    convergence_curve = zeros(max_iter)

    l = 0

    c1 = 0.18
    c2 = 0.82

    while l < max_iter
        for i = 1:npop
            Flag4ub = X[i, :] .> ub
            Flag4lb = X[i, :] .< lb
            X[i, :] = X[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness = objfun(X[i, :])

            if fitness < Best_score
                Best_score = fitness
                Best_pos = copy(X[i, :])
            end
        end

        convergence_curve[l+1] = Best_score

        a = 2 * (1 - (l / max_iter))
        Jump = 0.0
        dir = rand()

        for i = 1:npop
            r = rand()
            p = rand()

            if r >= 0.5
                if p > 0.18
                    Time = rand(dim)
                    sps = Best_pos ./ Time
                    Distance_S_Travel = sps .* Time
                    Distance_Fox_Rat[i, :] = 0.5 .* Distance_S_Travel
                    tt = sum(Time) / dim
                    t = tt / 2
                    Jump = 0.5 * 9.81 * t^2
                    X[i, :] = Distance_Fox_Rat[i, :] .* Jump * c1

                elseif p <= 0.18
                    Time = rand(dim)
                    sps = Best_pos ./ Time
                    Distance_S_Travel = sps .* Time
                    Distance_Fox_Rat[i, :] = 0.5 .* Distance_S_Travel
                    tt = sum(Time) / dim
                    t = tt / 2
                    Jump = 0.5 * 9.81 * t^2
                    X[i, :] = Distance_Fox_Rat[i, :] .* Jump * c2
                end

                if MinT > tt
                    MinT = tt
                end

            elseif r < 0.5
                X[i, :] = Best_pos .+ randn(dim) .* (MinT * a)
            end
        end

        l += 1
    end

    # return Best_score, Best_pos, convergence_curve
    return OptimizationResult(
        Best_pos,
        Best_score,
        convergence_curve)
end

function FOX(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return FOX(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end