"""
# References:

- Agushaka, Jeffrey O., Absalom E. Ezugwu, and Laith Abualigah.
  "Gazelle optimization algorithm: a novel nature-inspired metaheuristic optimizer."
  Neural Computing and Applications 35, no. 5 (2023): 4099-4131.
"""
function GazelleOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GazelleOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GazelleOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Top_gazelle_pos = zeros(dim)
    Top_gazelle_fit = Inf

    Convergence_curve = zeros(max_iter)
    stepsize = zeros(npop, dim)
    fitness = fill(Inf, npop)
    fit_old = fill(Inf, npop)

    gazelle = initialization(npop, dim, ub, lb)
    Prey_old = zeros(npop, dim)

    lb = repeat([lb], npop) .* ones(npop, dim)
    ub = repeat([ub], npop) .* ones(npop, dim)

    Iter = 0
    PSRs = 0.34
    S = 0.88

    while Iter < max_iter
        for i = 1:npop
            Flag4ub = gazelle[i, :] .> ub
            Flag4lb = gazelle[i, :] .< lb
            gazelle[i, :] .= gazelle[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = objfun(gazelle[i, :])

            if fitness[i] < Top_gazelle_fit
                Top_gazelle_fit = fitness[i]
                Top_gazelle_pos = gazelle[i, :]
            end
        end

        if Iter == 0
            fit_old = copy(fitness)
            Prey_old = copy(gazelle)
        end

        Inx = fit_old .< fitness
        gazelle .= Inx .* Prey_old .+ .~Inx .* gazelle
        fitness .= Inx .* fit_old .+ .~Inx .* fitness

        fit_old .= fitness
        Prey_old .= gazelle

        Elite = ones(npop, 1) * Top_gazelle_pos'
        CF = (1 - Iter / max_iter)^(2 * Iter / max_iter)

        # RL = 0.05 * levyFun(npop, dim, 1.5)  # Levy random number vector
        RL = 0.05 * levy(npop, dim, 1.5)  # Levy random number vector
        RB = randn(npop, dim)            # Brownian random number vector

        for i = 1:npop
            for j = 1:dim
                R = rand()
                r = rand()
                mu = ifelse(mod(Iter, 2) == 0, -1, 1)

                if r > 0.5
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] .- RB[i, j] .* gazelle[i, j])
                    gazelle[i, j] += rand() * R * stepsize[i, j]
                else
                    if i > npop / 2
                        stepsize[i, j] = RB[i, j] * (RL[i, j] * Elite[i, j] - gazelle[i, j])
                        gazelle[i, j] = Elite[i, j] + S * mu * CF * stepsize[i, j]
                    else
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] .- RL[i, j] .* gazelle[i, j])
                        gazelle[i, j] += S * mu * R * stepsize[i, j]
                    end
                end
            end
        end

        for i = 1:npop
            Flag4ub = gazelle[i, :] .> ub
            Flag4lb = gazelle[i, :] .< lb
            gazelle[i, :] .= gazelle[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = objfun(gazelle[i, :])

            if fitness[i] < Top_gazelle_fit
                Top_gazelle_fit = fitness[i]
                Top_gazelle_pos .= gazelle[i, :]
            end
        end

        Inx = fit_old .< fitness
        gazelle .= Inx .* Prey_old .+ .~Inx .* gazelle
        fitness .= Inx .* fit_old .+ .~Inx .* fitness

        fit_old .= fitness
        Prey_old .= gazelle

        if rand() < PSRs
            U = rand(npop, dim) .< PSRs
            gazelle .= gazelle .+ CF * ((lb .+ rand(npop, dim) .* (ub .- lb)) .* U)
        else
            r = rand()
            Rs = npop
            perm_indices1 = randperm(Rs)
            perm_indices2 = randperm(Rs)
            stepsize .= (PSRs * (1 - r) + r) * (gazelle[perm_indices1, :] .- gazelle[perm_indices2, :])
            gazelle .= gazelle .+ stepsize
        end

        Iter += 1
        Convergence_curve[Iter] = Top_gazelle_fit
    end

    # return Top_gazelle_fit, Top_gazelle_pos, Convergence_curve
    return OptimizationResult(
        Top_gazelle_pos,
        Top_gazelle_fit,
        Convergence_curve)
end

function GazelleOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GazelleOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

# function levyFun(n, m, beta)
#     num = gamma(1 + beta) * sin(pi * beta / 2)

#     den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)

#     sigma_u = (num / den)^(1 / beta)

#     u = rand(Normal(0, sigma_u), n, m)

#     v = rand(Normal(0, 1), n, m)

#     z = u ./ abs.(v) .^ (1 / beta)

#     return z
# end