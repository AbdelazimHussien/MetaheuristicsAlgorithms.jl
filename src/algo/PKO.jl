"""
Bouaouda, Anas, Fatma A. Hashim, Yassine Sayouti, and Abdelazim G. Hussien. 
"Pied kingfisher optimizer: a new bio-inspired algorithm for solving numerical optimization and industrial engineering problems." 
Neural Computing and Applications (2024): 1-59.
"""

function PKO(N::Int, Max_iter::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, dim::Int, fobj::Function)

    BF = 8  
    Crest_angles = 2 * π * rand()
    
    X = initialization(N, dim, ub, lb)
    
    Fitness = zeros(N)
    Convergence_curve = zeros(Max_iter,1)
    
    for i in 1:N
        Fitness[i] = fobj(X[i, :])
    end

    sorted_indexes = sortperm(Fitness)
    Best_position = X[sorted_indexes[1], :]
    Best_fitness = Fitness[sorted_indexes[1]]
    Convergence_curve[1] = Best_fitness

    t = 1
    PEmax = 0.5
    PEmin = 0

    while t < Max_iter + 1
        o = exp(-t / Max_iter)^2
        
        for i in 1:N
            if rand() < 0.8  
                j = i
                while i == j
                    seed = randperm(N)
                    j = seed[1]
                end

                beatingRate = rand() * (Fitness[j]) / Fitness[i]
                alpha = 2 * randn(dim) .- 1

                if rand() < 0.5
                    T = beatingRate - (t^(1 / BF) / Max_iter^(1 / BF))
                    X_1 = X[i, :] + alpha .* T .* (X[j, :] - X[i, :])
                else
                    T = (exp(1) - exp(((t - 1) / Max_iter)^(1 / BF))) * cos(Crest_angles)
                    X_1 = X[i, :] + alpha .* T .* (X[j, :] - X[i, :])
                end
            else  
                alpha = 2 * randn(dim) .- 1
                b = X[i, :] + o^2 * randn() .* Best_position
                HuntingAbility = rand() * (Fitness[i]) / Best_fitness
                X_1 = X[i, :] + HuntingAbility * o * alpha .* (b - Best_position)
            end

            FU = X_1 .> ub
            FL = X_1 .< lb
            X_1 = (X_1 .* .~(FU .+ FL)) + ub .* FU + lb .* FL
            
            fitnessn = fobj(X_1)

            if fitnessn < Fitness[i]
                Fitness[i] = fitnessn
                X[i, :] = X_1
            end

            if Fitness[i] < Best_fitness
                Best_fitness = Fitness[i]
                Best_position = X[i, :]
            end
        end

        PE = PEmax - (PEmax - PEmin) * (t / Max_iter)

        for i in 1:N
            alpha = 2 * randn(dim) .- 1
            if rand() > (1 - PE)
                X_1 = X[rand(1:N), :] + o * alpha .* abs.(X[i, :] - X[rand(1:N), :])
            else
                X_1 = X[i, :]
            end

            FU = X_1 .> ub
            FL = X_1 .< lb
            X_1 = (X_1 .* .~(FU .+ FL)) + ub .* FU + lb .* FL
            
            fitnessn = fobj(X_1)

            if fitnessn < Fitness[i]
                Fitness[i] = fitnessn
                X[i, :] = X_1
            end

            if Fitness[i] < Best_fitness
                Best_fitness = Fitness[i]
                Best_position = X[i, :]
            end
        end

        Convergence_curve[t] = Best_fitness
        t += 1
    end

    return Best_fitness, Best_position, Convergence_curve
end