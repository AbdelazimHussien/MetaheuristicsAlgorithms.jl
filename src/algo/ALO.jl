function Random_walk_around_antlion(Dim, max_iter, lb, ub, antlion, current_iter)
    
    if length(lb) == 1 && length(ub) == 1
        lb = fill(lb, Dim)
        ub = fill(ub, Dim)
    end

    if size(lb, 1) > size(lb, 2)
        lb = lb'
        ub = ub'
    end

    I = 1.0

    if current_iter > max_iter / 10
        I = 1 + 100 * (current_iter / max_iter)
    end

    if current_iter > max_iter / 2
        I = 1 + 1000 * (current_iter / max_iter)
    end

    if current_iter > max_iter * (3/4)
        I = 1 + 10000 * (current_iter / max_iter)
    end

    if current_iter > max_iter * 0.9
        I = 1 + 100000 * (current_iter / max_iter)
    end

    if current_iter > max_iter * 0.95
        I = 1 + 1000000 * (current_iter / max_iter)
    end

    # Decrease boundaries to converge towards the antlion
    lb = lb / I   # Equation (2.10) in the paper
    ub = ub / I   # Equation (2.11) in the paper

    # Move the interval of [lb, ub] around the antlion [lb + antlion, ub + antlion]
    if rand() < 0.5
        lb = lb .+ antlion   # Equation (2.8) in the paper
    else
        # lb .= -lb .+ antlion
        lb = -lb .+ antlion
    end

    if rand() >= 0.5
        ub = ub .+ antlion   # Equation (2.9) in the paper
    else
        ub = -ub .+ antlion
    end

    # Initialize the random walks matrix
    RWs = zeros(max_iter + 1, Dim)

    # Generate random walks and normalize them according to lb and ub
    for i in 1:Dim
        X = vcat(0, cumsum(2 * (rand(max_iter) .> 0.5) .- 1))  # Equation (2.1) in the paper
        a, b = minimum(X), maximum(X)
        c, d = lb[i], ub[i]
        # Normalize random walk to the interval [c, d]
        X_norm = ((X .- a) .* (d - c)) ./ (b - a) .+ c   # Equation (2.7) in the paper
        RWs[:, i] = X_norm
    end

    return RWs
end

function RouletteWheelSelectionF(weights)
    accumulation = cumsum(weights)  # Cumulative sum of weights
    p = rand() * accumulation[end]   # Random threshold based on the total weight
    chosen_index = findfirst(x -> x > p, accumulation)  # Find first index where accumulation exceeds p
    
    return chosen_index  # Return the chosen index
end


"""
# References:

- Mirjalili, Seyedali. "The ant lion optimizer." Advances in engineering software 83 (2015): 80-98.

"""
function ALO(N, Max_iter, lb, ub, dim, fobj)
    # Initialize the positions of antlions and ants
    antlion_position = initialization(N, dim, ub, lb)
    ant_position = initialization(N, dim, ub, lb)

    # Initialize variables to save the position of elite, sorted antlions, 
    # convergence curve, antlions fitness, and ants fitness
    Sorted_antlions = zeros(N, dim)
    Elite_antlion_position = zeros(dim)
    Elite_antlion_fitness = Inf
    Convergence_curve = zeros(Max_iter)
    antlions_fitness = zeros(N)
    ants_fitness = zeros(N)

    for i in axes(antlion_position, 1)
        antlions_fitness[i] = fobj(antlion_position[i, :])
    end

    sorted_indexes = sortperm(antlions_fitness)
    sorted_antlion_fitness = antlions_fitness[sorted_indexes]

    
    for newindex in 1:N
        Sorted_antlions[newindex, :] = antlion_position[sorted_indexes[newindex], :]
    end

    Elite_antlion_position .= Sorted_antlions[1, :]
    Elite_antlion_fitness = sorted_antlion_fitness[1]

    # Main loop starts from the second iteration
    Current_iter = 2
    while Current_iter <= Max_iter
        # Simulate random walks
        for i in axes(antlion_position, 1)
            # Select ant lions based on their fitness
            Rolette_index = RouletteWheelSelectionF(1 ./ sorted_antlion_fitness)
            Rolette_index = Rolette_index == -1 ? 1 : Rolette_index

            RA = Random_walk_around_antlion(dim, Max_iter, lb, ub, Sorted_antlions[Rolette_index, :], Current_iter)
            
            # RA is the random walk around the elite antlion
            RE = Random_walk_around_antlion(dim, Max_iter, lb, ub, Elite_antlion_position, Current_iter)

            ant_position[i, :] = (RA[Current_iter, :] + RE[Current_iter, :]) / 2
        end
        
        for i in axes(ant_position, 1)
            Flag4ub = ant_position[i, :] .> ub
            Flag4lb = ant_position[i, :] .< lb
            ant_position[i, :] = (ant_position[i, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb
            
            ants_fitness[i] = fobj(ant_position[i, :])
        end
        
        # Update antlion positions and fitnesses
        double_population = vcat(Sorted_antlions, ant_position)
        double_fitness = vcat(sorted_antlion_fitness, ants_fitness)
        
        I = sortperm(double_fitness)
        double_fitness_sorted = double_fitness[I]


        double_sorted_population = double_population[I, :]
        
        antlions_fitness .= double_fitness_sorted[1:N]
        Sorted_antlions .= double_sorted_population[1:N, :]

        if antlions_fitness[1] < Elite_antlion_fitness
            Elite_antlion_position .= Sorted_antlions[1, :]
            Elite_antlion_fitness = antlions_fitness[1]
        end
        
        # Keep the elite in the population
        Sorted_antlions[1, :] .= Elite_antlion_position
        antlions_fitness[1] = Elite_antlion_fitness
        
        # Update the convergence curve
        Convergence_curve[Current_iter] = Elite_antlion_fitness

        # Display the iteration and best optimum obtained so far
        # if Current_iter % 50 == 0
        #     println("At iteration $Current_iter, the elite fitness is $Elite_antlion_fitness")
        # end

        Current_iter += 1
    end

    return Elite_antlion_fitness, Elite_antlion_position, Convergence_curve
end