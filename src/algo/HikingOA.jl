"""
Oladejo, Sunday O., Stephen O. Ekwe, and Seyedali Mirjalili. 
"The Hiking Optimization Algorithm: A novel human-based metaheuristic approach." 
Knowledge-Based Systems 296 (2024): 111880.
"""

function HikingOA(hiker::Int, max_iter::Int, lb::Union{Real,AbstractVector}, ub::Union{Real,AbstractVector}, dim::Int, objfun)
    # Problem Parameters
    prob = objfun
    npop = hiker

    # Pre-allocate
    fit = zeros(npop)
    Best_iteration = zeros(max_iter + 1)

    # Initialize population
    Pop = [lb .+ (ub .- lb) .* rand(dim) for _ = 1:npop]

    # Evaluate initial fitness
    for q = 1:npop
        fit[q] = prob(Pop[q])
    end
    Best_iteration[1] = minimum(fit)

    # Main Loop
    for i = 1:max_iter
        _, ind = findmin(fit)
        Xbest = Pop[ind]

        for j = 1:npop
            Xini = Pop[j]

            theta = rand(0:50)
            s = tan(deg2rad(theta))
            SF = rand([1, 2])

            Vel = 6 * exp(-3.5 * abs(s + 0.05))
            newVel = Vel .+ rand(dim) .* (Xbest .- SF .* Xini)

            newPop = Xini .+ newVel
            newPop = clamp.(newPop, lb, ub)

            fnew = prob(newPop)
            if fnew < fit[j]
                Pop[j] = newPop
                fit[j] = fnew
            end
        end

        Best_iteration[i+1] = minimum(fit)
    end

    Best_Hike, idx = findmin(fit)
    Best_Position = Pop[idx]

    return Best_Hike, Best_Position, Best_iteration
end
