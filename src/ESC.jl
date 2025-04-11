"""
Ouyang, K., Fu, S., Chen, Y., Cai, Q., Heidari, A. A., & Chen, H. (2024). 
Escape: an optimization method based on crowd evacuation behaviors. 
Artificial Intelligence Review, 58(1), 19.
"""

function ESC(N::Int, maxIter::Int, lb::Int, ub::Int, dim::Int, fobj)


    if length(lb) == 1  && length(ub) == 1
        lb = fill(lb, dim)
        ub = fill(ub, dim)
    end


    if size(lb, 1) > size(lb, 2)
        lb = lb'
        ub = ub'
    end


    population = rand(N, dim) .* (ub .- lb) .+ lb
    best_fitness = Inf
    best_solution = zeros(dim)
    
    fitness = [fobj(population[i, :]) for i in 1:N]
    
    sorted = sortperm(fitness)
    population = population[sorted, :]
    fitness = fitness[sorted]
    fitness1 = fitness[1]
    eliteSize = 5
    fitness_history = zeros(maxIter)
    best_solutions = population[1:eliteSize, :]
    beta_base = 1.5
    t = 0
    mask_probability = 0.5

    while t < maxIter
        panicIndex = cos(pi / 2 * (t / (3 * maxIter)))
        sorted = sortperm(fitness)
        population = population[sorted, :]
        fitness = fitness[sorted]

        a = 0.15
        b = 0.35
        populationNew = copy(population)

        if t / maxIter <= 0.5
            calmCount = ceil(Int, round(a * N))
            conformCount = ceil(Int, round(b * N))
            calm = population[1:calmCount, :]
            conform = population[calmCount+1:calmCount+conformCount, :]
            panic = population[(calmCount + conformCount + 1):end, :]
            calmCenter = sum(calm, dims=1) / calmCount
            calmCenter = vec(calmCenter)
            
            for i in 1:N
                if size(panic, 1) > 0
                    randomPanicIndex = Int(floor(rand(1:size(panic, 1))))
                    panicIndividual = panic[randomPanicIndex, :]
                else
                    panicIndividual = zeros(1, dim)  
                end

                mask1 = rand(dim) .> mask_probability

                if i <= calmCount
                    minCalm = vec(minimum(calm, dims=1))
                    maxCalm = vec(maximum(calm, dims=1))
                    randVec = minCalm + rand(dim) .* (maxCalm - minCalm)
                    weight1 = adaptive_levy_weight(beta_base, dim, t, maxIter)

                    populationNew[i, :] .= population[i, :] 
                    .+ mask1 .* (weight1 .* (calmCenter - population[i, :]) + (randVec - population[i, :] .+ randn(dim) / 50)) .* panicIndex

                elseif i <= calmCount + conformCount
                    minConform = vec(minimum(conform, dims=1))
                    maxConform = vec(maximum(conform, dims=1))
                    randVec = minConform + rand(dim) .* (maxConform - minConform)
                    weight1 = adaptive_levy_weight(beta_base, dim, t, maxIter)
                    weight2 = adaptive_levy_weight(beta_base, dim, t, maxIter)
                    mask2 = rand(dim) .> mask_probability

                    populationNew[i, :] = population[i, :] .+ mask1 .* (weight1 .* (calmCenter - population[i, :]) + mask2 .* (weight2 .* (panicIndividual .- population[i, :])) + (randVec - population[i, :] .+ randn(dim) / 50) .* panicIndex)
                else
                    elite = best_solutions[rand(1:eliteSize), :]
                    randomInd = population[rand(1:N), :]#[1, :]
                    weight1 = adaptive_levy_weight(beta_base, dim, t, maxIter)
                    weight2 = adaptive_levy_weight(beta_base, dim, t, maxIter)
                    mask2 = rand(dim) .> mask_probability
                    randVec = elite .+ weight1 .* (randomInd - elite)
                    
                    populationNew[i, :] = population[i, :] .+ mask1 .* (weight1 .* (elite - population[i, :]) + mask2 .* (weight2 .* (randomInd - population[i, :])) + (randVec - population[i, :] .+ randn(dim) / 50) .* panicIndex)
                end
            end
        else
            for i in 1:N
                elite = best_solutions[rand(1:eliteSize), :]
                
                randInd = population[rand(1:N), :]
                weight1 = adaptive_levy_weight(beta_base, dim, t, maxIter)
                weight2 = adaptive_levy_weight(beta_base, dim, t, maxIter)
                mask1 = rand(dim) .> mask_probability
                mask2 = rand(dim) .> mask_probability
                
                populationNew[i, :] = population[i, :] .+ mask1 .* (weight1 .* (elite - population[i, :]) + mask2 .* (weight2 .* (randInd - population[i, :])))
            end
        end

        for i in 1:N
            populationNew[i, :] = max.(lb', min.(populationNew[i, :], ub'))
        end

        fitnessNew = [fobj(populationNew[i, :]) for i in 1:N]
        for i in 1:N
            if fitnessNew[i] < fitness[i]
                population[i, :] .= populationNew[i, :]
                fitness[i] = fitnessNew[i]
            end
        end

        sorted = sortperm(fitness)
        population = population[sorted, :]
        fitness = fitness[sorted]
        best_solutions = population[1:eliteSize, :]

        best_fitness = fitness[1]
        best_solution = population[1, :]
        t += 1
        fitness_history[t] = best_fitness
    end

    return best_fitness, best_solution, vcat([fitness1], fitness_history)
end

function adaptive_levy_weight(beta_base, dim, t, maxIter)
    beta = beta_base + 0.5 * sin(pi / 2 * t / maxIter)
    beta = clamp(beta, 0.1, 2.0)
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / 
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    u = randn(dim) * sigma^2
    v = randn(dim)
    w = abs.(u ./ abs.(v).^(1 / beta))
    return w ./ (maximum(w) + eps())
end