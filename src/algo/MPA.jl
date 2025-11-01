"""
# References:

- Faramarzi, Afshin, Mohammad Heidarinejad, Seyedali Mirjalili, and Amir H. Gandomi.
  "Marine Predators Algorithm: A nature-inspired metaheuristic."
  Expert systems with applications 152 (2020): 113377.
"""
function MPA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return MPA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function MPA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Top_predator_pos = zeros(dim)
    Top_predator_fit = Inf

    Convergence_curve = zeros(max_iter)
    stepsize = zeros(npop, dim)
    fitness = fill(Inf, npop, 1)

    fit_old = zeros(npop)

    Prey = initialization(npop, dim, ub, lb)
    Prey_old = zeros(size(Prey))

    lb = ones(npop, dim) .* lb
    ub = ones(npop, dim) .* ub

    Iter = 0
    FADs = 0.2
    P = 0.5

    while Iter < max_iter
        # Detecting top predator
        for i = 1:npop
            Flag4ub = Prey[i, :] .> ub
            Flag4lb = Prey[i, :] .< lb
            Prey[i, :] .= (Prey[i, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = objfun(Prey[i, :])

            if fitness[i] < Top_predator_fit
                Top_predator_fit = fitness[i]
                Top_predator_pos .= Prey[i, :]
            end
        end

        if Iter == 0
            fit_old = copy(fitness)
            Prey_old = copy(Prey)
        end

        Inx = fit_old .< fitness
        Indx = repeat(Inx, 1, dim)
        Prey .= Indx .* Prey_old .+ .~Indx .* Prey
        fitness .= Inx .* fit_old .+ .~Inx .* fitness

        fit_old = copy(fitness)
        Prey_old = copy(Prey)

        Elite = repeat(reshape(Top_predator_pos, 1, dim), npop, 1)
        CF = (1 - Iter / max_iter)^(2 * Iter / max_iter)

        RL = 0.05 * levy(npop, dim, 1.5)
        RB = randn(npop, dim)

        for i = 1:npop
            for j = 1:dim
                R = rand()

                if Iter < max_iter / 3
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] += P * R * stepsize[i, j]

                elseif Iter > max_iter / 3 && Iter < 2 * max_iter / 3
                    if i > npop / 2
                        stepsize[i, j] = RB[i, j] * (RB[i, j] * Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                    else
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] += P * R * stepsize[i, j]
                    end

                else
                    stepsize[i, j] = RL[i, j] * (RL[i, j] * Elite[i, j] - Prey[i, j])
                    Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                end
            end
        end

        for i = 1:npop
            Flag4ub = Prey[i, :] .> ub
            Flag4lb = Prey[i, :] .< lb
            Prey[i, :] .= (Prey[i, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = objfun(Prey[i, :])

            if fitness[i] < Top_predator_fit
                Top_predator_fit = fitness[i]
                Top_predator_pos .= Prey[i, :]
            end
        end

        if Iter == 0
            fit_old = copy(fitness)
            Prey_old = copy(Prey)
        end

        Inx = fit_old .< fitness
        Indx = repeat(Inx, 1, dim)
        Prey .= Indx .* Prey_old .+ .~Indx .* Prey
        fitness .= Inx .* fit_old .+ .~Inx .* fitness

        fit_old = copy(fitness)
        Prey_old = copy(Prey)

        if rand() < FADs
            U = rand(npop, dim) .< FADs
            Prey += CF * ((lb .+ rand(npop, dim) .* (ub - lb)) .* U)
        else
            r = rand()
            Rs = npop
            stepsize = (FADs * (1 - r) + r) * (Prey[shuffle(1:Rs), :] - Prey[shuffle(1:Rs), :])
            Prey += stepsize
        end

        Iter += 1
        Convergence_curve[Iter] = Top_predator_fit
    end

    # return Top_predator_fit, Top_predator_pos, Convergence_curve
    return OptimizationResult(
        Top_predator_pos,
        Top_predator_fit,
        Convergence_curve)
end

function MPA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return MPA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end