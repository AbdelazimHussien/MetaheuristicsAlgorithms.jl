"""
Dhiman, Gaurav, Meenakshi Garg, Atulya Nagar, Vijay Kumar, and Mohammad Dehghani. 
"A novel algorithm for global optimization: rat swarm optimizer." 
Journal of Ambient Intelligence and Humanized Computing 12 (2021): 8457-8482.
"""
function RSO(npop::Int, max_iter::Int, lb::Union{Real,AbstractVector}, ub::Union{Real,AbstractVector}, dim::Int, objfun)
    Position = zeros(dim)
    Score = Inf
    Positions = initialization(npop, dim, ub, lb)
    Convergence = zeros(max_iter)

    l = 0
    x = 1
    y = 5
    R = floor(Int, (y - x) * rand() + x)

    while l < max_iter
        # Flag4Upper_bound = Positions .> ub
        # Flag4Lower_bound = Positions .< lb
        Positions .= max.(lb, min.(Positions, ub))  # Clamp values between bounds

        fitness_values = [objfun(Positions[i, :]) for i in axes(Positions, 1)]

        best_index = argmin(fitness_values)
        best_fitness = fitness_values[best_index]

        if best_fitness < Score
            Score = best_fitness
            Position .= Positions[best_index, :]
        end

        A = R - l * (R / max_iter)

        for i in axes(Positions, 1)
            for j in axes(Positions, 2)
                C = 2 * rand()
                P_vec = A * Positions[i, j] + abs(C * (Position[j] - Positions[i, j]))
                P_final = Position[j] - P_vec
                Positions[i, j] = P_final
            end
        end

        l += 1
        Convergence[l] = Score
    end

    return Score, Position, Convergence
end
