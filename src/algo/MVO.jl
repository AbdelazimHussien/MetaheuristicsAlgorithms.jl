
function RouletteWheelSelection(weights)
    accumulation = cumsum(weights)  # Cumulative sum of weights
    p = rand() * accumulation[end]   # Random value in the range of total weights
    chosen_index = -1

    # Loop through the accumulated weights using eachindex
    for index in eachindex(accumulation)
        if accumulation[index] > p
            chosen_index = index
            break
        end
    end

    return chosen_index
end

"""
# References:

- Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Abdolreza Hatamlou.
  "Multi-verse optimizer: a nature-inspired algorithm for global optimization."
  Neural Computing and Applications 27 (2016): 495-513.
"""
function MVO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return MVO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function MVO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)

    # Initialize the best universe and its inflation rate (fitness)
    Best_universe = zeros(1, dim)
    Best_universe_Inflation_rate = Inf

    # Initialize the positions of universes
    Universes = initialization(npop, dim, ub, lb)

    # Minimum and maximum of Wormhole Existence Probability (WEP)
    WEP_Max = 1
    WEP_Min = 0.2

    Convergence_curve = zeros(max_iter)

    # Iteration (time) counter
    Time = 1

    # Main loop
    while Time <= max_iter
        # Eq. (3.3) in the paper: Calculate WEP
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / max_iter)

        # Eq. (3.4) in the paper: Calculate Travelling Distance Rate (TDR)
        TDR = 1 - ((Time)^(1 / 6) / (max_iter)^(1 / 6))

        # Initialize inflation rates (fitness values)
        Inflation_rates = zeros(npop)

        for i = 1:npop
            # Boundary checking
            Flag4ub = Universes[i, :] .> ub
            Flag4lb = Universes[i, :] .< lb
            Universes[i, :] = (Universes[i, :] .* .~(Flag4ub .+ Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb

            # Calculate inflation rate (fitness) for each universe
            Inflation_rates[i] = objfun(Universes[i, :])

            # Elitism: Update the best universe
            if Inflation_rates[i] < Best_universe_Inflation_rate
                Best_universe_Inflation_rate = Inflation_rates[i]
                Best_universe = copy(Universes[i, :])
            end
        end

        # Sort inflation rates and get the sorted universes
        sorted_Inflation_rates, sorted_indexes = sort(Inflation_rates), sortperm(Inflation_rates)
        Sorted_universes = Universes[sorted_indexes, :]

        # Normalized inflation rates
        normalized_sorted_Inflation_rates = normalize(sorted_Inflation_rates)

        # Keep the elite universe
        Universes[1, :] = Sorted_universes[1, :]

        # Update the position of universes
        for i = 2:npop # Start from 2, since the first is the elite
            for j = 1:dim
                r1 = rand()
                if r1 < normalized_sorted_Inflation_rates[i]
                    White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates)
                    White_hole_index = White_hole_index == -1 ? 1 : White_hole_index
                    Universes[i, j] = Sorted_universes[White_hole_index, j]
                end

                # Eq. (3.2) if boundaries are the same
                if length(lb) == 1
                    r2 = rand()
                    if r2 < WEP
                        r3 = rand()
                        if r3 < 0.5
                            Universes[i, j] = Best_universe[j] + TDR * ((ub - lb) * rand() + lb)
                        else
                            Universes[i, j] = Best_universe[j] - TDR * ((ub - lb) * rand() + lb)
                        end
                    end
                else
                    # Eq. (3.2) if bounds differ for each variable
                    r2 = rand()
                    if r2 < WEP
                        r3 = rand()
                        if r3 < 0.5
                            Universes[i, j] = Best_universe[j] + TDR * ((ub[j] - lb[j]) * rand() + lb[j])
                        else
                            Universes[i, j] = Best_universe[j] - TDR * ((ub[j] - lb[j]) * rand() + lb[j])
                        end
                    end
                end
            end
        end

        # Update the convergence curve
        Convergence_curve[Time] = Best_universe_Inflation_rate

        # Print best universe details every 50 iterations
        # if mod(Time, 50) == 0
        #     println("At iteration $Time, the best universe fitness is $Best_universe_Inflation_rate")
        # end

        Time += 1
    end

    # return Best_universe_Inflation_rate, Best_universe, Convergence_curve
    return OptimizationResult(
        Best_universe,
        Best_universe_Inflation_rate,
        Convergence_curve)
end

function MVO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return MVO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end