
function exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)
    if rand() < p1
        current_vulture_X = random_vulture_X - abs.((2 * rand()) * random_vulture_X - current_vulture_X) * F
    else
        current_vulture_X = random_vulture_X .- F .+ rand() * ((upper_bound - lower_bound) .* rand() .+ lower_bound)
    end
    return current_vulture_X
end

function exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound)
    
    # Phase 1
    if abs(F) < 0.5
        if rand() < p2
            A = Best_vulture1_X - ((Best_vulture1_X .* current_vulture_X) ./ (Best_vulture1_X .- current_vulture_X .^ 2)) * F
            B = Best_vulture2_X - ((Best_vulture2_X .* current_vulture_X) ./ (Best_vulture2_X .- current_vulture_X .^ 2)) * F
            current_vulture_X = (A + B) / 2
        else
            current_vulture_X = random_vulture_X - abs.(random_vulture_X - current_vulture_X) .* F .* levyFlight(variables_no)
        end
    end

    # Phase 2
    if abs(F) >= 0.5
        if rand() < p3
            current_vulture_X = abs.(2 * rand() * random_vulture_X - current_vulture_X) * (F + rand()) - (random_vulture_X - current_vulture_X)
        else
            s1 = random_vulture_X .* (rand() * current_vulture_X / (2 * π)) .* cos.(current_vulture_X)
            s2 = random_vulture_X .* (rand() * current_vulture_X / (2 * π)) .* sin.(current_vulture_X)
            current_vulture_X = random_vulture_X - (s1 + s2)
        end
    end

    return current_vulture_X
end

function levyFlight(d)
    beta = 3 / 2

    sigma = (gamma(1 + beta) * sin(π * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    u = randn(d) * sigma
    v = randn(d)
    step = u ./ abs.(v).^(1 / beta)

    return step
end

function random_select(Best_vulture1_X, Best_vulture2_X, alpha, betha)
    probabilities = [alpha, betha]

    # Implement roulette wheel selection
    selected_index = rouletteWheelSelection(probabilities)
    
    if selected_index == 1
        random_vulture_X = Best_vulture1_X
    else
        random_vulture_X = Best_vulture2_X
    end

    return random_vulture_X
end

function rouletteWheelSelection(x)
    random_value = rand()
    cumulative_sum = cumsum(x)
    
    # Find the first index where cumulative sum is greater than or equal to random_value
    index = findfirst(c -> c >= random_value, cumulative_sum)
    
    return index
end



"""
# References:

- Abdollahzadeh, B., Gharehchopogh, F. S., & Mirjalili, S. (2021). African vultures optimization algorithm: A new nature-inspired metaheuristic algorithm for global optimization problems.  Computers & Industrial Engineering, 158, 107408.

"""
function AVOA(pop_size, max_iter, lower_bound, upper_bound, variables_no, fobj)

    # Initialize Best vultures
    Best_vulture1_X = zeros(variables_no)
    Best_vulture1_F = Inf
    Best_vulture2_X = zeros(variables_no)
    Best_vulture2_F = Inf

    # Initialize the first random population of vultures
    X = initialization(pop_size, variables_no, upper_bound, lower_bound)

    # Controlling parameters
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    alpha = 0.8
    betha = 0.2
    gamma = 2.5

    # Main loop
    current_iter = 0
    convergence_curve = zeros(max_iter)

    while current_iter < max_iter
        # Evaluate the fitness of the population
        # for i in 1:size(X, 1)
        for i in axes(X, 1)
            current_vulture_X = X[i, :]
            current_vulture_F = fobj(current_vulture_X)

            # Update the best vultures if needed
            if current_vulture_F < Best_vulture1_F
                Best_vulture1_F = current_vulture_F
                Best_vulture1_X = current_vulture_X
            elseif current_vulture_F > Best_vulture1_F && current_vulture_F < Best_vulture2_F
                Best_vulture2_F = current_vulture_F
                Best_vulture2_X = current_vulture_X
            end
        end

        a = rand(-2:0.001:2) * ((sin((π / 2) * (current_iter / max_iter))^gamma) + cos((π / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * rand() + 1) * (1 - (current_iter / max_iter)) + a


        for i in axes(X, 1)
            current_vulture_X = X[i, :]
            F = P1 * (2 * rand() - 1)

            random_vulture_X = random_select(Best_vulture1_X, Best_vulture2_X, alpha, betha)

            if abs(F) >= 1
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)
            else
                current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound)
            end

            X[i, :] = current_vulture_X
        end

        current_iter += 1
        convergence_curve[current_iter] = Best_vulture1_F

        X = clamp.(X, lower_bound, upper_bound)
    end

    return Best_vulture1_F, Best_vulture1_X, convergence_curve
end
