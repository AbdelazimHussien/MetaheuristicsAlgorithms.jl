"""
# References:

- Braik, Malik, Alaa Sheta, and Heba Al-Hiary. "A novel meta-heuristic search algorithm for solving optimization problems: capuchin search algorithm." Neural computing and applications 33, no. 7 (2021): 2515-2547.

"""
function CoatiOA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

    lowerbound = ones(dimension) .* lowerbound  
    upperbound = ones(dimension) .* upperbound  

    # INITIALIZATION
    X = zeros(SearchAgents, dimension)
    for i = 1:dimension
        X[:, i] = lowerbound[i] .+ rand(SearchAgents) .* (upperbound[i] - lowerbound[i])  
    end

    fit = [fitness(X[i, :]) for i = 1:SearchAgents]

    Xbest = zeros(dimension)
    fbest = Inf

    best_so_far = 0

    for t = 1:Max_iterations
        best, location = findmin(fit)
        if t == 1 || best < fbest
            fbest = best
            Xbest = X[location, :]
        end

        X_P1 = zeros(SearchAgents, dimension)
        F_P1 = zeros(SearchAgents)
        for i = 1:div(SearchAgents, 2)
            iguana = Xbest
            I = round(Int, 1 + rand())
            X_P1[i, :] = X[i, :] .+ rand() .* (iguana - I .* X[i, :])  # Eq. (4)
            X_P1[i, :] = clamp.(X_P1[i, :], lowerbound, upperbound)

            L = X_P1[i, :]
            F_P1[i] = fitness(L)
            if F_P1[i] < fit[i]
                X[i, :] = X_P1[i, :]
                fit[i] = F_P1[i]
            end
        end

        for i = div(SearchAgents, 2) + 1:SearchAgents
            iguana = lowerbound .+ rand(dimension) .* (upperbound - lowerbound)  # Eq (5)
            F_HL = fitness(iguana)
            I = round(Int, 1 + rand())

            if fit[i] > F_HL
                X_P1[i, :] = X[i, :] .+ rand() .* (iguana - I .* X[i, :])  # Eq. (6)
            else
                X_P1[i, :] = X[i, :] .+ rand() .* (X[i, :] - iguana)  # Eq. (6)
            end
            X_P1[i, :] = clamp.(X_P1[i, :], lowerbound, upperbound)

            L = X_P1[i, :]
            F_P1[i] = fitness(L)
            if F_P1[i] < fit[i]
                X[i, :] = X_P1[i, :]
                fit[i] = F_P1[i]
            end
        end

        X_P2 = zeros(SearchAgents, dimension)
        F_P2 = zeros(SearchAgents)
        for i = 1:SearchAgents
            LO_LOCAL = lowerbound ./ t 
            HI_LOCAL = upperbound ./ t  

            X_P2[i, :] = X[i, :] .+ (1 - 2 .* rand()) .* (LO_LOCAL .+ rand() .* (HI_LOCAL - LO_LOCAL))  # Eq. (8)
            X_P2[i, :] = clamp.(X_P2[i, :], LO_LOCAL, HI_LOCAL)

            L = X_P2[i, :]
            F_P2[i] = fitness(L)
            if F_P2[i] < fit[i]
                X[i, :] = X_P2[i, :]
                fit[i] = F_P2[i]
            end
        end

        best_so_far = fbest
        average = mean(fit)
    end

    return fbest, Xbest, best_so_far
end