"""
Agushaka, Jeffrey O., Absalom E. Ezugwu, and Laith Abualigah. 
"Gazelle optimization algorithm: a novel nature-inspired metaheuristic optimizer." 
Neural Computing and Applications 35, no. 5 (2023): 4099-4131.
"""

using Random
using Distributions
using SpecialFunctions  # for gamma function


function GazelleOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    # Top_gazelle_pos = zeros(1, dim)
    Top_gazelle_pos = zeros(dim)
    Top_gazelle_fit = Inf

    Convergence_curve = zeros(Max_iter)
    stepsize = zeros(SearchAgents_no, dim)
    fitness = fill(Inf, SearchAgents_no)
    fit_old = fill(Inf, SearchAgents_no)

    gazelle = initialization(SearchAgents_no, dim, ub, lb)
    Prey_old = zeros(SearchAgents_no, dim)
    
    Xmin = repeat([lb], SearchAgents_no) .* ones(SearchAgents_no, dim)
    Xmax = repeat([ub], SearchAgents_no) .* ones(SearchAgents_no, dim)

    Iter = 0
    PSRs = 0.34
    S = 0.88

    while Iter < Max_iter
        # Evaluate top gazelle
        for i in 1:SearchAgents_no
            Flag4ub = gazelle[i, :] .> ub
            Flag4lb = gazelle[i, :] .< lb
            gazelle[i, :] .= gazelle[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = fobj(gazelle[i, :])

            if fitness[i] < Top_gazelle_fit
                Top_gazelle_fit = fitness[i]
                Top_gazelle_pos = gazelle[i, :]
            end
        end

        # Update fitness values and positions
        if Iter == 0
            fit_old = copy(fitness)
            Prey_old = copy(gazelle)
        end

        Inx = fit_old .< fitness
        gazelle .= Inx .* Prey_old .+ .~Inx .* gazelle
        fitness .= Inx .* fit_old .+ .~Inx .* fitness

        fit_old .= fitness
        Prey_old .= gazelle

        # Calculate Elite and CF
        # Elite = repeat([Top_gazelle_pos], SearchAgents_no) .* ones(SearchAgents_no, dim)
        # Elite = repeat([Top_gazelle_pos], SearchAgents_no, 1)
        # Elite = repeat([Top_gazelle_pos'], 1, SearchAgents_no)
        # Elite = (Top_gazelle_pos .* ones(SearchAgents_no, length(Top_gazelle_pos)))'
        Elite = ones(SearchAgents_no, 1) * Top_gazelle_pos'
        # Elite = hcat([Top_gazelle_pos for _ in 1:SearchAgents_no]...)
        # Elite = Top_gazelle_pos .* ones(SearchAgents_no, dim)
        # println("Top_gazelle_pos ", size(Top_gazelle_pos))
        # println("Elite ", size(Elite))
        CF = (1 - Iter / Max_iter)^(2 * Iter / Max_iter)

        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)  # Levy random number vector
        # println("RLLLL ", size(RL))
        RB = randn(SearchAgents_no, dim)            # Brownian random number vector

        # Exploitation and Exploration
        for i in 1:SearchAgents_no
            for j in 1:dim
                R = rand()
                r = rand()
                mu = ifelse(mod(Iter, 2) == 0, -1, 1)

                if r > 0.5
                    # stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * gazelle[i, j])
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] .- RB[i, j] .* gazelle[i, j])
                    gazelle[i, j] += rand() * R * stepsize[i, j]
                else
                    if i > SearchAgents_no / 2
                        stepsize[i, j] = RB[i, j] * (RL[i, j] * Elite[i, j] - gazelle[i, j])
                        gazelle[i, j] = Elite[i, j] + S * mu * CF * stepsize[i, j]
                    else
                        # stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * gazelle[i, j])
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] .- RL[i, j] .* gazelle[i, j])
                        gazelle[i, j] += S * mu * R * stepsize[i, j]
                    end
                end
            end
        end

        # Update fitness and positions after adjustment
        for i in 1:SearchAgents_no
            Flag4ub = gazelle[i, :] .> ub
            Flag4lb = gazelle[i, :] .< lb
            gazelle[i, :] .= gazelle[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = fobj(gazelle[i, :])

            if fitness[i] < Top_gazelle_fit
                Top_gazelle_fit = fitness[i]
                Top_gazelle_pos .= gazelle[i, :]
            end
        end

        # Update fitness values and prey positions
        Inx = fit_old .< fitness
        gazelle .= Inx .* Prey_old .+ .~Inx .* gazelle
        fitness .= Inx .* fit_old .+ .~Inx .* fitness

        fit_old .= fitness
        Prey_old .= gazelle

        # Apply PSRs
        if rand() < PSRs
            U = rand(SearchAgents_no, dim) .< PSRs
            gazelle .= gazelle .+ CF * ((Xmin .+ rand(SearchAgents_no, dim) .* (Xmax .- Xmin)) .* U)
        else
            r = rand()
            Rs = SearchAgents_no
            perm_indices1 = randperm(Rs)
            perm_indices2 = randperm(Rs)
            stepsize .= (PSRs * (1 - r) + r) * (gazelle[perm_indices1, :] .- gazelle[perm_indices2, :])
            gazelle .= gazelle .+ stepsize
        end

        Iter += 1
        Convergence_curve[Iter] = Top_gazelle_fit
    end

    return Top_gazelle_fit, Top_gazelle_pos, Convergence_curve
end

function levy(n, m, beta)
    # Numerator for sigma_u calculation
    num = gamma(1 + beta) * sin(pi * beta / 2)
    
    # Denominator for sigma_u calculation
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)

    # Standard deviation
    sigma_u = (num / den)^(1 / beta)

    # Generate u from Normal distribution with mean 0 and std sigma_u
    u = rand(Normal(0, sigma_u), n, m)

    # Generate v from Normal distribution with mean 0 and std 1
    v = rand(Normal(0, 1), n, m)

    # Levy random number
    z = u ./ abs.(v).^(1 / beta)

    return z
end