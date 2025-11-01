"""

# References:

- Abualigah, Laith, Ali Diabat, Seyedali Mirjalili, Mohamed Abd Elaziz, and Amir H. Gandomi. "The arithmetic optimization algorithm." Computer methods in applied mechanics and engineering 376 (2021): 113609.
"""
function AOArithmetic(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return AOArithmetic(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function AOArithmetic(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb) # AOArithmetic(npop, max_iter, lb, ub, dim, Chung_Reynolds)

    # Two variables to keep the positions and the fitness value of the best-obtained solution
    Best_P = zeros(dim)
    Best_FF = Inf
    Conv_curve = zeros(max_iter)

    # Initialize the positions of solution
    X = initialization(npop, dim, ub, lb)
    Xnew = copy(X)
    Ffun = zeros(npop)  # (fitness values)
    Ffun_new = zeros(npop)  # (fitness values)

    MOP_Max = 1
    MOP_Min = 0.2
    C_Iter = 1
    Alpha = 5
    Mu = 0.499

    for i = 1:npop
        Ffun[i] = objfun(X[i, :])  # Calculate the fitness values of solutions
        if Ffun[i] < Best_FF
            Best_FF = Ffun[i]
            Best_P = X[i, :]
        end
    end

    while C_Iter <= max_iter  # Main loop
        MOP = 1 - ((C_Iter)^(1 / Alpha) / (max_iter)^(1 / Alpha))  # Probability Ratio
        MOA = MOP_Min + C_Iter * ((MOP_Max - MOP_Min) / max_iter)  # Accelerated function

        # Update the Position of solutions
        for i = 1:npop
            for j = 1:dim
                r1 = rand()
                if length(lb) == 1
                    if r1 < MOA
                        r2 = rand()
                        if r2 > 0.5
                            Xnew[i, j] = Best_P[j] / (MOP + eps()) * ((ub - lb) * Mu + lb)
                        else
                            Xnew[i, j] = Best_P[j] * MOP * ((ub - lb) * Mu + lb)
                        end
                    else
                        r3 = rand()
                        if r3 > 0.5
                            Xnew[i, j] = Best_P[j] - MOP * ((ub - lb) * Mu + lb)
                        else
                            Xnew[i, j] = Best_P[j] + MOP * ((ub - lb) * Mu + lb)
                        end
                    end
                else
                    r1 = rand()
                    if r1 < MOA
                        r2 = rand()
                        if r2 > 0.5
                            Xnew[i, j] = Best_P[j] / (MOP + eps()) * ((ub[j] - lb[j]) * Mu + lb[j])
                        else
                            Xnew[i, j] = Best_P[j] * MOP * ((ub[j] - lb[j]) * Mu + lb[j])
                        end
                    else
                        r3 = rand()
                        if r3 > 0.5
                            Xnew[i, j] = Best_P[j] - MOP * ((ub[j] - lb[j]) * Mu + lb[j])
                        else
                            Xnew[i, j] = Best_P[j] + MOP * ((ub[j] - lb[j]) * Mu + lb[j])
                        end
                    end
                end
            end

            # Check boundaries
            Flag_UB = Xnew[i, :] .> ub  # Check if they exceed upper boundaries
            Flag_LB = Xnew[i, :] .< lb  # Check if they exceed lower boundaries
            Xnew[i, :] = max.(min.(Xnew[i, :], ub), lb) # (Xnew[i, :] .* .!((Flag_UB .+ Flag_LB))) .+ ub .* Flag_UB .+ lb .* Flag_LB

            Ffun_new[i] = objfun(Xnew[i, :])  # Calculate Fitness function
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

    # return Best_FF, Best_P, Conv_curve
    return OptimizationResult(
        Best_FF,
        Best_P,
        Conv_curve)
end

function AOArithmetic(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return AOArithmetic(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end