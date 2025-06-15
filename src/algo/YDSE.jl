"""
Abdel-Basset, Mohamed, Doaa El-Shahat, Mohammed Jameel, and Mohamed Abouhawwash.
"Young’s double-slit experiment optimizer: A novel metaheuristic optimization algorithm for global and constraint optimization problems." 
Computer Methods in Applied Mechanics and Engineering 403 (2023): 115652.
"""


function YDSE(NP, Max_iter, LB, UB, Dim, F_obj)
    L = 1       
    d = 5e-3    
    I = 0.01    
    Lambda = 5e-6 
    Delta = 0.38 

    S = initialization(NP, Dim, UB, LB) 

    FS = zeros(NP, Dim)
    SS = zeros(NP, Dim)
    S_Mean = mean(S, dims=1) |> vec
    for i in 1:NP
        FS[i, :] = S[i, :] .+ I .* (2 * rand() - 1) .* (S_Mean .- S[i, :]) 
        SS[i, :] = S[i, :] .- I .* (2 * rand() - 1) .* (S_Mean .- S[i, :]) 
    end


    BestFringe = zeros(Dim) 
    Best_Fringe_fitness = Inf
    Fringes_fitness = zeros(NP)
    X = zeros(NP, Dim)

    for i in 1:NP
        if i == 1
            Delta_L = 0 
        elseif isodd(i)
            m = i - 1
            Delta_L = (2 * m + 1) * Lambda / 2 
        else
            m = i - 1
            Delta_L = m * Lambda 
        end
        X[i, :] = (FS[i, :] .+ SS[i, :]) ./ 2 .+ Delta_L 
    end

    X .= clamp.(X, LB, UB)

    for i in 1:NP
        Fringes_fitness[i] = F_obj(X[i, :])
    end

    sorted_indices = sortperm(Fringes_fitness)
    Fringes_fitness = Fringes_fitness[sorted_indices]
    X = X[sorted_indices, :]
    Best_Fringe_fitness = Fringes_fitness[1]
    BestFringe = X[1, :]

    Int_max0 = 1e-20
    cgcurve = zeros(Max_iter)
    iter = 1
    X_New = zeros(NP, Dim)

    while iter <= Max_iter
        cgcurve[iter] = Best_Fringe_fitness

        sorted_indices = sortperm(Fringes_fitness)
        Fringes_fitness = Fringes_fitness[sorted_indices]
        X = X[sorted_indices, :]

        q = iter / Max_iter 
        Int_max = Int_max0 * q 

        for i in 1:NP
            a = iter^(2 * rand() - 1) 
            H = 2 .* rand(Dim) .- 1
            z = a ./ H 
            r1 = rand()

            if i == 1
                beta = q * cosh(pi / iter) 
                A_bright = 2 / (1 + sqrt(abs(1 - beta^2))) 
                A = 1:2:NP
                randomIndex = rand(A)
                X_New[i, :] = BestFringe .+ Int_max .* A_bright .* X[i, :] .- r1 .* z .* X[randomIndex, :] 
            elseif isodd(i)
                beta = q * cosh(pi / iter) 
                A_bright = 2 / (1 + sqrt(abs(1 - beta^2))) 
                m = i - 1
                y_bright = Lambda * L * m / d 
                Int_bright = Int_max .* cos((pi * d) / (Lambda * L) * y_bright).^2 
                s = randperm(NP)[1:2]
                g = 2 * rand() - 1
                Y = X[s[2], :] .- X[s[1], :] 
                X_New[i, :] = X[i, :] .- ((1 - g) .* A_bright .* Int_bright .* X[i, :] .+ g .* Y) 
            else
                A_dark = Delta * atanh(-(q) + 1) 
                m = i - 1
                y_dark = Lambda * L * (m + 0.5) / d 
                Int_dark = Int_max .* cos((pi * d) / (Lambda * L) * y_dark).^2 
                X_New[i, :] = X[i, :] .- (r1 .* A_dark .* Int_dark .* X[i, :] .- z .* BestFringe) 
            end

            X_New[i, :] = clamp.(X_New[i, :], LB, UB)
        end

        X_New_Fitness = zeros(NP)
        for i in 1:NP
            X_New_Fitness[i] = F_obj(X_New[i, :])
            if X_New_Fitness[i] < Fringes_fitness[i]
                X[i, :] = X_New[i, :]
                Fringes_fitness[i] = X_New_Fitness[i]
                if Fringes_fitness[i] < Best_Fringe_fitness
                    Best_Fringe_fitness = Fringes_fitness[i]
                    BestFringe = X[i, :]
                end
            end
        end

        iter += 1
    end

    return Best_Fringe_fitness, BestFringe, cgcurve
end