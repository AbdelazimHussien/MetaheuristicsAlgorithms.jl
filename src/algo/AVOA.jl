
function exploration(current_vulture_X, random_vulture_X, F, p1, ub, lb)
    if rand() < p1
        current_vulture_X = random_vulture_X - abs.((2 * rand()) * random_vulture_X - current_vulture_X) * F
    else
        current_vulture_X = random_vulture_X .- F .+ rand() * ((ub - lb) .* rand() .+ lb)
    end
    return current_vulture_X
end

function exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, dim, ub, lb)

    # Phase 1
    if abs(F) < 0.5
        if rand() < p2
            A = Best_vulture1_X - ((Best_vulture1_X .* current_vulture_X) ./ (Best_vulture1_X .- current_vulture_X .^ 2)) * F
            B = Best_vulture2_X - ((Best_vulture2_X .* current_vulture_X) ./ (Best_vulture2_X .- current_vulture_X .^ 2)) * F
            current_vulture_X = (A + B) / 2
        else
            # current_vulture_X = random_vulture_X - abs.(random_vulture_X - current_vulture_X) .* F .* levyFlight(dim)
            current_vulture_X = random_vulture_X - abs.(random_vulture_X - current_vulture_X) .* F .* levy(dim)
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
function AVOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return AVOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function AVOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)

    # Initialize Best vultures
    Best_vulture1_X = zeros(dim)
    Best_vulture1_F = Inf
    Best_vulture2_X = zeros(dim)
    Best_vulture2_F = Inf

    # Initialize the first random population of vultures
    X = initialization(npop, dim, ub, lb)

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
            current_vulture_F = objfun(current_vulture_X)

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
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, ub, lb)
            else
                current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, dim, ub, lb)
            end

            X[i, :] = current_vulture_X
        end

        current_iter += 1
        convergence_curve[current_iter] = Best_vulture1_F

        X = clamp.(X, lb', ub')
    end

    # return Best_vulture1_F, Best_vulture1_X, convergence_curve
    return OptimizationResult(
        Best_vulture1_X,
        Best_vulture1_F,
        convergence_curve)
end

function AVOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return AVOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end