"""
Trojovský, Pavel, and Mohammad Dehghani. 
"Pelican optimization algorithm: A novel nature-inspired algorithm for engineering applications." 
Sensors 22, no. 3 (2022): 855.
"""


function POA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fobj)
    lowerbound = ones(dimension) * lowerbound
    upperbound = ones(dimension) * upperbound

    X = initialization(SearchAgents, dimension, upperbound, lowerbound)

    
    fit = [fobj(X[i, :]) for i in 1:SearchAgents]
    
    fbest = Inf
    Xbest = []

    best_so_far = zeros(Max_iterations)
    average = zeros(Max_iterations)
    
    for t in 1:Max_iterations
        best, location = findmin(fit)
        if t == 1 || best < fbest
            fbest = best
            Xbest = X[location, :]
        end
        
        k = randperm(SearchAgents)[1]
        X_FOOD = X[k, :]
        F_FOOD = fit[k]

        for i in 1:SearchAgents
            I = round(Int, 1 + rand())
            if fit[i] > F_FOOD
                X_new = X[i, :] .+ rand() .* (X_FOOD - I .* X[i, :])
            else
                X_new = X[i, :] .+ rand() .* (X[i, :] - X_FOOD)
            end
            X_new = clamp.(X_new, lowerbound, upperbound)

            f_new = fobj(X_new)
            if f_new <= fit[i]
                X[i, :] = X_new
                fit[i] = f_new
            end

            X_new = X[i, :] .+ 0.2 * (1 - t / Max_iterations) .* (2 * rand(dimension) .- 1) .* X[i, :]
            X_new = clamp.(X_new, lowerbound, upperbound)

            f_new = fobj(X_new)
            if f_new <= fit[i]
                X[i, :] = X_new
                fit[i] = f_new
            end
        end
        
        best_so_far[t] = fbest
        # println("best_so_far[$t]  ", fbest)
        average[t] = mean(fit)
    end

    return fbest, Xbest, best_so_far
end