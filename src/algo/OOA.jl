"""
# References:

- Dehghani, Mohammad, and Pavel Trojovský.
  "Osprey optimization algorithm: A new bio-inspired metaheuristic algorithm for solving engineering optimization problems."
  Frontiers in Mechanical Engineering 8 (2023): 1126450.
"""
function OOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return OOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function OOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # Lower limit for variables
    lb = ones(dim) .* lb
    # Upper limit for variables
    ub = ones(dim) .* ub

    # INITIALIZATION
    X = zeros(npop, dim)  # Initialize the population matrix
    IndividualsPos = initialization(npop, dim, lb, ub)
    # for i = 1:dim
    #     X[:, i] = lb[i] .+ rand(npop) .* (ub[i] - lb[i])  # Initial population
    # end

    fit = zeros(npop)  # Initialize fitness array
    for i = 1:npop
        L = X[i, :]
        fit[i] = objfun(L)  # Compute fitness for each individual
    end

    best_so_far = zeros(max_iter)  # Array to store best scores
    average = zeros(max_iter)  # Array to store average scores
    fbest = Inf
    xbest = zeros(dim)

    # Main loop for iterations
    for t = 1:max_iter
        # Update: BEST proposed solution
        Fbest, blocation = findmin(fit)  # Get the minimum fitness value and its index

        # Update best solution found
        if t == 1
            xbest = X[blocation, :]  # Optimal location
            fbest = Fbest  # The optimization objective function
        elseif Fbest < fbest
            fbest = Fbest
            xbest = X[blocation, :]
        end

        # Loop through each search agent
        for i = 1:npop
            # PHASE 1: POSITION IDENTIFICATION AND HUNTING THE FISH (EXPLORATION)
            fish_position = findall(x -> x < fit[i], fit)  # Get indices of fitter fish

            if isempty(fish_position)
                selected_fish = xbest
            else
                if rand() < 0.5
                    selected_fish = xbest
                else
                    k = rand(1:length(fish_position))  # Randomly select one of the fitter fish
                    selected_fish = X[fish_position[k]]
                end
            end

            I = round(Int, 1 + rand())  # Randomly choose 1 or 2
            X_new_P1 = X[i, :] + rand() * (selected_fish .- I * X[i, :])  # Eq(5)
            X_new_P1 = max.(X_new_P1, lb)  # Clip to lower bound
            X_new_P1 = min.(X_new_P1, ub)  # Clip to upper bound

            # Update position based on Eq (6)
            L = X_new_P1
            fit_new_P1 = objfun(L)
            if fit_new_P1 < fit[i]
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1
            end

            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            X_new_P1 = X[i, :] + (lb .+ rand() * (ub .- lb)) / t  # Eq(7)
            X_new_P1 = max.(X_new_P1, lb)  # Clip to lower bound
            X_new_P1 = min.(X_new_P1, ub)  # Clip to upper bound

            # Update position based on Eq (8)
            L = X_new_P1
            fit_new_P1 = objfun(L)
            if fit_new_P1 < fit[i]
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1
            end
        end

        best_so_far[t] = fbest
        average[t] = mean(fit)  # Calculate average fitness
        # println("It $t = $fbest")
    end

    # return fbest, xbest, best_so_far
    return OptimizationResult(
        xbest,
        fbest,
        best_so_far)
end

function OOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return OOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end