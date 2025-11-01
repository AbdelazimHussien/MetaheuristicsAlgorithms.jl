"""
# References:

- Braik, Malik, Abdelaziz Hammouri, Jaffar Atwan, Mohammed Azmi Al-Betar, and Mohammed A. Awadallah.
  "White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm for global optimization problems."
  Knowledge-Based Systems 243 (2022): 108457.
"""
function WSO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return WSO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function WSO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, whiteSharks::Integer, max_iter::Integer)
    dim = length(lb)
    ccurve = zeros(max_iter)

    WSO_Positions = initialization(whiteSharks, dim, ub, lb)

    v = zeros(size(WSO_Positions))

    fit = [objfun(WSO_Positions[i, :]) for i = 1:whiteSharks]

    fitness = copy(fit)

    fmin0, index = findmin(fit)

    wbest = copy(WSO_Positions)
    gbest = copy(WSO_Positions[index, :])

    fmax = 0.75
    fmin = 0.07
    tau = 4.11
    mu = 2 / abs(2 - tau - sqrt(tau^2 - 4 * tau))
    pmin, pmax = 0.5, 1.5
    a0, a1, a2 = 6.250, 100, 0.0005

    for ite = 1:max_iter
        mv = 1 / (a0 + exp((max_iter / 2.0 - ite) / a1))
        s_s = abs((1 - exp(-a2 * ite / max_iter)))

        p1 = pmax + (pmax - pmin) * exp(-(4 * ite / max_iter)^2)
        p2 = pmin + (pmax - pmin) * exp(-(4 * ite / max_iter)^2)

        nu = rand(1:whiteSharks, whiteSharks)

        for i = 1:whiteSharks
            rmin, rmax = 1.0, 3.0
            rr = rmin + rand() * (rmax - rmin)
            wr = abs(((2 * rand()) - (1 * rand() + rand())) / rr)
            v[i, :] = mu * v[i, :] + wr * (wbest[nu[i], :] - WSO_Positions[i, :])
        end

        for i = 1:whiteSharks
            f = fmin + (fmax - fmin) / (fmax + fmin)

            a = sign.(WSO_Positions[i, :] .- ub) .> 0
            b = sign.(WSO_Positions[i, :] .- lb) .< 0

            wo = .~(a .& b)

            if rand() < mv
                WSO_Positions[i, :] = WSO_Positions[i, :] .* .!wo + ub .* a + lb .* b
            else
                WSO_Positions[i, :] = WSO_Positions[i, :] + v[i, :] / f
            end
        end

        for i = 1:whiteSharks
            for j = 1:dim
                if rand() < s_s
                    Dist = abs(rand() * (gbest[j] - 1 * WSO_Positions[i, j]))

                    if i == 1
                        WSO_Positions[i, j] = gbest[j] + rand() * Dist * sign(rand() - 0.5)
                    else
                        WSO_Pos = copy(WSO_Positions)
                        WSO_Pos[i, j] = gbest[j] + rand() * Dist * sign(rand() - 0.5)
                        WSO_Positions[i, j] = (WSO_Pos[i, j] + WSO_Positions[i-1, j]) / 2 * rand()
                    end
                end
            end
        end

        for i = 1:whiteSharks
            if all(WSO_Positions[i, :] .>= lb) && all(WSO_Positions[i, :] .<= ub)
                fit[i] = objfun(WSO_Positions[i, :])

                if fit[i] < fitness[i]
                    wbest[i, :] = WSO_Positions[i, :]
                    fitness[i] = fit[i]
                end

                if fitness[i] < fmin0
                    fmin0 = fitness[i]
                    gbest = copy(wbest[index, :])
                end
            end
        end
        ccurve[ite] = fmin0
    end

    # return fmin0, gbest, ccurve
    return OptimizationResult(
        gbest,
        fmin0,
        ccurve)
end

function WSO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return WSO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end