"""
# References:

- Abualigah, Laith, Mohamed Abd Elaziz, Putra Sumari, Zong Woo Geem, and Amir H. Gandomi.
  "Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic optimizer."
  Expert Systems with Applications 191 (2022): 116158.
"""
function RSA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return RSA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function RSA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Best_P = zeros(dim)
    Best_F = Inf
    X = initialization(npop, dim, ub, lb)
    Xnew = zeros(npop, dim)
    Conv = zeros(max_iter)

    t = 1
    Alpha = 0.1
    Beta = 0.005
    Ffun = zeros(npop)
    Ffun_new = zeros(npop)

    for i in 1:npop
        Ffun[i] = objfun(X[i, :])
        if Ffun[i] < Best_F
            Best_F = Ffun[i]
            Best_P = X[i, :]
        end
    end

    while t <= max_iter
        ES = 2 * randn() * (1 - (t / max_iter))
        for i in 2:npop
            for j in 1:dim
                R = (Best_P[j] - X[rand(1:npop), j]) / (Best_P[j] + eps())
                P = Alpha + (X[i, j] - mean(X[i, :])) / (Best_P[j] * (ub - lb) + eps())
                Eta = Best_P[j] * P

                if t < max_iter / 4
                    Xnew[i, j] = Best_P[j] - Eta * Beta - R * rand()
                elseif t < 2 * max_iter / 4
                    Xnew[i, j] = Best_P[j] * X[rand(1:npop), j] * ES * rand()
                elseif t < 3 * max_iter / 4
                    Xnew[i, j] = Best_P[j] * P * rand()
                else
                    Xnew[i, j] = Best_P[j] - Eta * eps() - R * rand()
                end
            end

            Flag_UB = Xnew[i, :] .> ub
            Flag_LB = Xnew[i, :] .< lb
            Xnew[i, :] = (Xnew[i, :] .* .! (Flag_UB .| Flag_LB)) .+ (ub .* Flag_UB) .+ (lb .* Flag_LB)
            Ffun_new[i] = objfun(Xnew[i, :])

            if Ffun_new[i] < Ffun[i]
                X[i, :] = Xnew[i, :]
                Ffun[i] = Ffun_new[i]
            end

            if Ffun[i] < Best_F
                Best_F = Ffun[i]
                Best_P = X[i, :]
            end
        end

        Conv[t] = Best_F

        # if t % 50 == 0  # Print the best universe details after every 50 iterations
        #      println("At iteration $t the best solution fitness is $Best_F")
        # end

        t += 1
    end

    # return Best_F, Best_P, Conv
    return OptimizationResult(
        Best_P,
        Best_F,
        Conv)
end

function RSA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return RSA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end