function levyF(Popsize, Dim, beta)
    sigma_u = (gamma(1 + beta) * sin(π * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    u = randn(Popsize, Dim) * sigma_u
    v = randn(Popsize, Dim)
    step = u ./ abs.(v).^(1 / beta)
    return step
end


"""
# References:

- Zhao, Shijie, Tianran Zhang, Shilin Ma, and Miao Chen. 
    "Dandelion Optimizer: A nature-inspired metaheuristic algorithm for engineering applications." 
    Engineering Applications of Artificial Intelligence 114 (2022): 105075.
    
"""
function DO(Popsize, Maxiteration, LB, UB, Dim, Fobj)
    dandelions = rand(Popsize, Dim) .* (UB - LB) .+ LB
    dandelionsFitness = zeros(Popsize)
    Convergence_curve = zeros(Maxiteration)

    for i in 1:Popsize
        dandelionsFitness[i] = Fobj(dandelions[i, :])
    end
    
    sorted_indexes = sortperm(dandelionsFitness)
    Best_position = dandelions[sorted_indexes[1], :]
    Best_fitness = dandelionsFitness[sorted_indexes[1]]
    Convergence_curve[1] = Best_fitness
    
    t = 2
    
    while t <= Maxiteration
        beta = randn(Popsize, Dim)
        alpha = rand() * ((1 / Maxiteration^2) * t^2 - 2 / Maxiteration * t + 1)
        a = 1 / (Maxiteration^2 - 2 * Maxiteration + 1)
        b = -2 * a
        c = 1 - a - b
        k = 1 - rand() * (c + a * t^2 + b * t)
        
        if randn() < 1.5
            dandelions_1 = copy(dandelions)
            for i in 1:Popsize
                lamb = abs.(randn(Dim))
                theta = (2 * rand() - 1) * π
                row = 1 / exp(theta)
                vx = row * cos(theta)
                vy = row * sin(theta)
                NEW = rand(Dim) .* (UB - LB) .+ LB
                dandelions_1[i, :] .= dandelions[i, :] .+ alpha .* vx .* vy .* logpdf.(Normal(0, 1), lamb) .* (NEW .- dandelions[i, :])
            end
        else
            dandelions_1 = copy(dandelions)
            for i in 1:Popsize
                dandelions_1[i, :] .= dandelions[i, :] .* k
            end
        end
        dandelions .= dandelions_1
        dandelions = clamp.(dandelions, LB, UB)
        
        dandelions_mean = sum(dandelions, dims=1) ./ Popsize
        dandelions_2 = copy(dandelions)
        for i in 1:Popsize
            for j in 1:Dim
                dandelions_2[i, j] = dandelions[i, j] - beta[i, j] * alpha * (dandelions_mean[1, j] - beta[i, j] * alpha * dandelions[i, j])
            end
        end
        dandelions .= dandelions_2
        dandelions = clamp.(dandelions, LB, UB)
        
        Step_length = levyF(Popsize, Dim, 1.5)
        Elite = repeat(Best_position', Popsize, 1)
        dandelions_3 = copy(dandelions)
        for i in 1:Popsize
            for j in 1:Dim
                dandelions_3[i, j] = Elite[i, j] + Step_length[i, j] * alpha * (Elite[i, j] - dandelions[i, j] * (2 * t / Maxiteration))
            end
        end
        dandelions .= dandelions_3
        dandelions = clamp.(dandelions, LB, UB)
        
        for i in 1:Popsize
            dandelionsFitness[i] = Fobj(dandelions[i, :])
        end

        sorted_indexes = sortperm(dandelionsFitness)
        dandelions = dandelions[sorted_indexes, :]
        SortfitbestN = dandelionsFitness[sorted_indexes]
        

        if SortfitbestN[1] < Best_fitness
            Best_position = dandelions[1, :]
            Best_fitness = SortfitbestN[1]
        end
        
        Convergence_curve[t] = Best_fitness
        t += 1
    end
    
    return Best_fitness, Best_position, Convergence_curve
end
