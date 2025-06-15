"""

# References:

- Abualigah, Laith, Ali Diabat, Seyedali Mirjalili, Mohamed Abd Elaziz, and Amir H. Gandomi. "The arithmetic optimization algorithm." Computer methods in applied mechanics and engineering 376 (2021): 113609.

"""
function AOArithmetic(N, M_Iter, LB, UB, Dim, F_obj) # AOArithmetic(SearchAgents_no, Max_iteration, lb, ub, dim, Chung_Reynolds)
    
    # Two variables to keep the positions and the fitness value of the best-obtained solution
    Best_P = zeros(Dim)
    Best_FF = Inf
    Conv_curve = zeros(M_Iter)

    # Initialize the positions of solution
    X = initialization(N, Dim, UB, LB)
    Xnew = copy(X)
    Ffun = zeros(N)  # (fitness values)
    Ffun_new = zeros(N)  # (fitness values)

    MOP_Max = 1
    MOP_Min = 0.2
    C_Iter = 1
    Alpha = 5
    Mu = 0.499

    for i in 1:N
        Ffun[i] = F_obj(X[i, :])  # Calculate the fitness values of solutions
        if Ffun[i] < Best_FF
            Best_FF = Ffun[i]
            Best_P = X[i, :]
        end
    end

    while C_Iter <= M_Iter  # Main loop
        MOP = 1 - ((C_Iter)^(1/Alpha) / (M_Iter)^(1/Alpha))  # Probability Ratio
        MOA = MOP_Min + C_Iter * ((MOP_Max - MOP_Min) / M_Iter)  # Accelerated function

        # Update the Position of solutions
        for i in 1:N
            for j in 1:Dim
                r1 = rand()
                if length(LB) == 1
                    if r1 < MOA
                        r2 = rand()
                        if r2 > 0.5
                            Xnew[i, j] = Best_P[j] / (MOP + eps()) * ((UB - LB) * Mu + LB)
                        else
                            Xnew[i, j] = Best_P[j] * MOP * ((UB - LB) * Mu + LB)
                        end
                    else
                        r3 = rand()
                        if r3 > 0.5
                            Xnew[i, j] = Best_P[j] - MOP * ((UB - LB) * Mu + LB)
                        else
                            Xnew[i, j] = Best_P[j] + MOP * ((UB - LB) * Mu + LB)
                        end
                    end
                else
                    r1 = rand()
                    if r1 < MOA
                        r2 = rand()
                        if r2 > 0.5
                            Xnew[i, j] = Best_P[j] / (MOP + eps()) * ((UB[j] - LB[j]) * Mu + LB[j])
                        else
                            Xnew[i, j] = Best_P[j] * MOP * ((UB[j] - LB[j]) * Mu + LB[j])
                        end
                    else
                        r3 = rand()
                        if r3 > 0.5
                            Xnew[i, j] = Best_P[j] - MOP * ((UB[j] - LB[j]) * Mu + LB[j])
                        else
                            Xnew[i, j] = Best_P[j] + MOP * ((UB[j] - LB[j]) * Mu + LB[j])
                        end
                    end
                end
            end

            # Check boundaries
            Flag_UB = Xnew[i, :] .> UB  # Check if they exceed upper boundaries
            Flag_LB = Xnew[i, :] .< LB  # Check if they exceed lower boundaries
            Xnew[i, :] = max.(min.(Xnew[i, :], UB), LB) # (Xnew[i, :] .* .!((Flag_UB .+ Flag_LB))) .+ UB .* Flag_UB .+ LB .* Flag_LB

            Ffun_new[i] = F_obj(Xnew[i, :])  # Calculate Fitness function
            if Ffun_new[i] < Ffun[i]
                X[i, :] = Xnew[i, :]
                Ffun[i] = Ffun_new[i]
            end
            if Ffun[i] < Best_FF
                Best_FF = Ffun[i]
                Best_P = X[i, :]
            end
        end

        # Update the convergence curve
        Conv_curve[C_Iter] = Best_FF

        # Print the best solution details after every 50 iterations
        # if C_Iter % 50 == 0
        #     println("At iteration ", C_Iter, " the best solution fitness is ", Best_FF)
        # end

        C_Iter += 1  # Incremental iteration
    end

    return Best_FF, Best_P, Conv_curve
end