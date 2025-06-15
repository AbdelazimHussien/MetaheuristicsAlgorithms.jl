"""
Su, Hang, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, and Huiling Chen. 
"RIME: A physics-based optimization." 
Neurocomputing 532 (2023): 183-214.
"""


function RIME(N::Int, Max_iter::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, dim::Int, fobj)
    println("RIME is now tackling your problem")

    Best_rime = zeros(dim)
    Best_rime_rate = Inf  
    Rimepop = initialization(N, dim, ub, lb)  
    Lb = fill(lb, dim)  
    Ub = fill(ub, dim)  
    it = 1  
    Convergence_curve = zeros(Max_iter)
    Rime_rates = zeros(N)  
    newRime_rates = zeros(N)
    W = 5  

    for i in 1:N
        Rime_rates[i] = fobj(Rimepop[i, :])  
        if Rime_rates[i] < Best_rime_rate
            Best_rime_rate = Rime_rates[i]
            Best_rime = Rimepop[i, :]
        end
    end

    while it <= Max_iter
        RimeFactor = (rand() - 0.5) * 2 * cos(pi * it / (Max_iter / 10)) * (1 - round(it * W / Max_iter) / W)  # Eq.(3),(4),(5)
        E = sqrt(it / Max_iter)  
        newRimepop = copy(Rimepop)  
        normalized_rime_rates = normalize(Rime_rates)  

        for i in 1:N
            for j in 1:dim
                r1 = rand()
                if r1 < E
                    newRimepop[i, j] = Best_rime[j] + RimeFactor * ((Ub[j] - Lb[j]) * rand() + Lb[j])  # Eq.(3)
                end
                r2 = rand()
                if r2 < normalized_rime_rates[i]
                    newRimepop[i, j] = Best_rime[j]  
                end
            end
        end

        for i in 1:N
            newRimepop[i, :] = clamp.(newRimepop[i, :], lb, ub)
            newRime_rates[i] = fobj(newRimepop[i, :])

            if newRime_rates[i] < Rime_rates[i]
                Rime_rates[i] = newRime_rates[i]
                Rimepop[i, :] = newRimepop[i, :]
                if newRime_rates[i] < Best_rime_rate
                    Best_rime_rate = Rime_rates[i]
                    Best_rime = Rimepop[i, :]
                end
            end
        end

        Convergence_curve[it] = Best_rime_rate
        println("Convergence_curve[$it] = $Best_rime_rate")
        it += 1
    end

    return Best_rime_rate, Best_rime, Convergence_curve
end