"""
Ezugwu, Absalom E., Jeffrey O. Agushaka, Laith Abualigah, Seyedali Mirjalili, and Amir H. Gandomi. 
"Prairie dog optimization algorithm." 
Neural Computing and Applications 34, no. 22 (2022): 20017-20065.
"""


function levym(n, m, beta)
    num = gamma(1 + beta) * sin(pi * beta / 2)  
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)  
    sigma_u = (num / den)^(1 / beta)  

    u = rand(Normal(0, sigma_u), n, m)  
    v = rand(Normal(0, 1), n, m)  

    z = u ./ (abs.(v) .^ (1 / beta))  
    return z
end

function PDO(N, T, LB, UB, Dim, F_obj)
    PDBest_P = zeros(Dim)        
    Best_PD = Inf                
    X = initialization(N, Dim, UB, LB)  
    Xnew = zeros(N, Dim)
    PDConv = zeros(T)            
    M = Dim                      
    t = 1                        
    rho = 0.005                  
    epsPD = 0.1                  
    OBest = zeros(N)             
    CBest = zeros(N)             

    for i in 1:N
        Flag_UB = X[i, :] .> UB  
        Flag_LB = X[i, :] .< LB  
        X[i, :] .= (X[i, :] .* .~(Flag_UB .| Flag_LB)) .+ UB .* Flag_UB .+ LB .* Flag_LB

        OBest[i] = F_obj(X[i, :])  
        if OBest[i] < Best_PD
            Best_PD = OBest[i]
            PDBest_P .= X[i, :]
        end
    end

    while t <= T
        mu = ifelse(t % 2 == 0, -1, 1)
        DS = 1.5 * randn() * (1 - t / T)^(2 * t / T) * mu
        PE = 1.5 * (1 - t / T)^(2 * t / T) * mu
        RL = levym(N, Dim, 1.5)  
        TPD = repeat(PDBest_P', N)  

        for i in 1:N
            for j in 1:M
                cpd = rand() * ((TPD[i, j] - X[rand(1:N), j]) / (TPD[i, j] + eps()))
                P = rho + (X[i, j] - mean(X[i, :])) / (TPD[i, j] * (UB - LB) + eps())
                eCB = PDBest_P[j] * P

                if t < T / 4
                    Xnew[i, j] = PDBest_P[j] - eCB * epsPD - cpd * RL[i, j]
                elseif t < 2 * T / 4
                    Xnew[i, j] = PDBest_P[j] * X[rand(1:N), j] * DS * RL[i, j]
                elseif t < 3 * T / 4
                    Xnew[i, j] = PDBest_P[j] * PE * rand()
                else
                    Xnew[i, j] = PDBest_P[j] - eCB * eps() - cpd * rand()
                end
            end

            Flag_UB = Xnew[i, :] .> UB
            Flag_LB = Xnew[i, :] .< LB
            Xnew[i, :] .= (Xnew[i, :] .* .~(Flag_UB .| Flag_LB)) .+ UB .* Flag_UB .+ LB .* Flag_LB

            CBest[i] = F_obj(Xnew[i, :])
            if CBest[i] < OBest[i]
                X[i, :] .= Xnew[i, :]
                OBest[i] = CBest[i]
            end
            if OBest[i] < Best_PD
                Best_PD = OBest[i]
                PDBest_P .= X[i, :]
            end
        end

        PDConv[t] = Best_PD

        t += 1
    end

    return Best_PD, PDBest_P, PDConv
end