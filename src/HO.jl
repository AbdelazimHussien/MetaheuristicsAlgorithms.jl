"""
Amiri, M.H., Mehrabi Hashjin, N., Montazeri, M., Mirjalili, S. and Khodadadi, N., 2024. 
Hippopotamus optimization algorithm: a novel nature-inspired optimization algorithm. 
Scientific Reports, 14(1), p.5032.
"""

using Random

function HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

    # println("Hi")

    # Lower and upper limits for variables
    lowerbound = fill(lowerbound, dimension)
    upperbound = fill(upperbound, dimension)

    # Initialization
    X = initialization(SearchAgents, dimension, lowerbound, upperbound)
    fit = zeros(SearchAgents)
    for i in 1:SearchAgents
        fit[i] = fitness(X[i, :])          # Directly assign to fit[i]
        # println("Hi $(fitness(X[i, :]'))")
    end
    Xbest, fbest = nothing, Inf

    # Main Loop
    best_so_far = zeros(Max_iterations)
    for t in 1:Max_iterations
        # Update the Best Candidate Solution
        best, location = findmin(fit)
        if t == 1 || best < fbest
            Xbest = X[location, :]
            fbest = best
        end

        # Phase 1: Hippopotamuses position update in the river or pond (Exploration)
        for i in 1:(SearchAgents ÷ 2)
            Dominant_hippopotamus = Xbest
            I1, I2 = rand(Bool) ? 1 : 2, rand(Bool) ? 1 : 2
            Ip1 = rand(Bool, 2)
            RandGroupNumber = rand(1:SearchAgents)
            # RandGroup = randperm(SearchAgents, RandGroupNumber)
            # RandGroup = rand(1:SearchAgents, RandGroupNumber; replace=false)
            RandGroup = sample(1:SearchAgents, RandGroupNumber; replace=false)
            # println("RandGroup ", RandGroup)

            # Mean of Random Group
            MeanGroup = mean(X[RandGroup, :], dims=1)
            # println("1 - size(MeanGroup) ", size(MeanGroup))
            # MeanGroup .= MeanGroup * (length(RandGroup) != 1) .+ X[RandGroup[1], :] * (length(RandGroup) == 1)
            # MeanGroup = MeanGroup * (length(RandGroup) != 1) .+ X[RandGroup[1], :] * (length(RandGroup) == 1)
            # MeanGroup = MeanGroup .* (length(RandGroup) != 1) .+ X[RandGroup[1], :] * (length(RandGroup) == 1)
            if length(RandGroup) != 1
                MeanGroup = mean(X[RandGroup, :], dims=1)
            else
                MeanGroup = X[RandGroup[1], :]
            end
            # println("11 - size(MeanGroup) ", size(MeanGroup))

            # Alpha array definition
            Alfa = [
                I2 * rand(dimension) .+ .~Ip1[1],
                2 * rand(dimension) .- 1,
                rand(dimension),
                I1 * rand(dimension) .+ .~Ip1[2],
                rand()
            ]

            A, B = Alfa[rand(1:5)], Alfa[rand(1:5)]
            X_P1 = X[i, :] .+ rand() .* (Dominant_hippopotamus .- I1 .* X[i, :])
            T = exp(-t / Max_iterations)

            # println("size(Dominant_hippopotamus) ", size(Dominant_hippopotamus))
            # println("size(MeanGroup) ", size(MeanGroup))
            # println("size(A) ", size(A), " size(B) ", size(B))

            MeanGroup = vec(MeanGroup)

            if T > 0.6
                X_P2 = X[i, :] .+ A .* (Dominant_hippopotamus .- I2 .* MeanGroup)
            else
                if rand() > 0.5
                    X_P2 = X[i, :] .+ B .* (MeanGroup .- Dominant_hippopotamus)
                else
                    X_P2 = (upperbound .- lowerbound) .* rand(dimension) .+ lowerbound
                end
            end
            X_P2 .= clamp.(X_P2, lowerbound, upperbound)

            ###
            # println("size(X_P1) ", size(X_P1))
            # println("size(X_P2) ", size(X_P2))

            F_P1, F_P2 = fitness(X_P1), fitness(X_P2)
            if F_P1 < fit[i]
                X[i, :] = X_P1
                fit[i] = F_P1
            end
            if F_P2 < fit[i]
                X[i, :] = X_P2
                fit[i] = F_P2
            end
        end

        # Phase 2: Hippopotamus defense against predators (Exploration)
        for i in (SearchAgents ÷ 2 + 1):SearchAgents
            predator = lowerbound .+ rand(dimension) .* (upperbound .- lowerbound)
            F_HL = fitness(predator)
            distance2Leader = abs.(predator .- X[i, :])

            b, c, d, l = rand(2.0:4.0), rand(1.0:1.5), rand(2.0:3.0), rand(-2 * π:2 * π)
            RL = 0.05 * levy(SearchAgents, dimension, 1.5)

            if fit[i] > F_HL
                X_P3 = RL[i, :] .* predator .+ (b / (c - d * cos(l))) .* (1 ./ distance2Leader)
            else
                X_P3 = RL[i, :] .* predator .+ (b / (c - d * cos(l))) .* (1 ./ (2 .* distance2Leader .+ rand(dimension)))
            end
            X_P3 .= clamp.(X_P3, lowerbound, upperbound)

            F_P3 = fitness(X_P3)
            if F_P3 < fit[i]
                X[i, :] = X_P3
                fit[i] = F_P3
            end
        end

        # Phase 3: Hippopotamus Escaping from the Predator (Exploitation)
        for i in 1:SearchAgents
            LO_LOCAL = lowerbound ./ t
            HI_LOCAL = upperbound ./ t
            # D = [rand(dimension) * 2 - 1, rand(), randn()][rand(1:3)]
            D = [rand(dimension) * 2 .- 1, rand(), randn()][rand(1:3)]
            X_P4 = X[i, :] .+ rand() .* (LO_LOCAL .+ D .* (HI_LOCAL .- LO_LOCAL))
            X_P4 .= clamp.(X_P4, lowerbound, upperbound)

            F_P4 = fitness(X_P4)
            if F_P4 < fit[i]
                X[i, :] = X_P4
                fit[i] = F_P4
            end
        end

        best_so_far[t] = fbest
        # println("Iteration $t: Best Cost = $fbest")
    end

    return fbest, Xbest, best_so_far
end
