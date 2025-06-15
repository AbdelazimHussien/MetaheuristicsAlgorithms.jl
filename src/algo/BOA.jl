"""
# References:

- Arora, Sankalap, and Satvir Singh. "Butterfly optimization algorithm: a novel approach for global optimization." Soft computing 23 (2019): 715-734.

"""
function BOA(n, N_iter, Lb, Ub, dim, fobj)
    p = 0.6                  
    power_exponent = 0.1      
    sensory_modality = 0.01   

    Sol = initialization(n, dim, Ub, Lb)
    Fitness = zeros(n)
    for i in 1:n
        Fitness[i] = fobj(Sol[i, :])
    end

    fmin, I = findmin(Fitness)
    best_pos = Sol[I, :]

    S = copy(Sol)
    Convergence_curve = zeros(N_iter)

    for t in 1:N_iter
        for i in 1:n  
            Fnew = fobj(S[i, :])
            FP = sensory_modality * (Fnew^power_exponent)

            
            if rand() < p
                dis = rand() * rand() * best_pos - Sol[i, :]  # Eq. (2) in paper
                S[i, :] = Sol[i, :] + dis * FP
            else
                
                epsilon = rand()
                JK = randperm(n)
                dis = epsilon * epsilon * Sol[JK[1], :] - Sol[JK[2], :]
                S[i, :] = Sol[i, :] + dis * FP  # Eq. (3) in paper
            end

            
            Flag4Ub = S[i, :] .> Ub
            Flag4Lb = S[i, :] .< Lb
            S[i, :] = (S[i, :] .* .~(Flag4Ub .+ Flag4Lb)) .+ Ub .* Flag4Ub .+ Lb .* Flag4Lb

            
            Fnew = fobj(S[i, :])  

            if Fnew <= Fitness[i]
                Sol[i, :] = S[i, :]
                Fitness[i] = Fnew
            end

            if Fnew <= fmin
                best_pos = S[i, :]
                fmin = Fnew
            end
        end

        Convergence_curve[t] = fmin

        sensory_modality = sensory_modality_NEW(sensory_modality, N_iter)
    end

    return fmin, best_pos, Convergence_curve
end

function sensory_modality_NEW(x, Ngen)
    return x + (0.025 / (x * Ngen))
end

# Initialization of agents within bounds
function initialization(n, dim, Ub, Lb)
    return rand(n, dim) .* (Ub - Lb) .+ Lb
end
