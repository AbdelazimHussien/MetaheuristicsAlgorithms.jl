"""
# References:

- Abdel-Basset, Mohamed, Doaa El-Shahat, Mohammed Jameel, and Mohamed Abouhawwash.
  "Young's double-slit experiment optimizer: A novel metaheuristic optimization algorithm for global and constraint optimization problems."
  Computer Methods in Applied Mechanics and Engineering 403 (2023): 115652.
"""
function YDSE(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return YDSE(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function YDSE(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    L = 1

    d = 5e-3
    I = 0.01
    Lambda = 5e-6
    Delta = 0.38

    S = initialization(npop, dim, ub, lb)

    FS = zeros(npop, dim)
    SS = zeros(npop, dim)
    S_Mean = mean(S, dims=1) |> vec
    for i = 1:npop
        FS[i, :] = S[i, :] .+ I .* (2 * rand() - 1) .* (S_Mean .- S[i, :])
        SS[i, :] = S[i, :] .- I .* (2 * rand() - 1) .* (S_Mean .- S[i, :])
    end

    best_fringe = zeros(dim)
    best_fringe_fitness = Inf
    Fringes_fitness = zeros(npop)
    X = zeros(npop, dim)

    for i = 1:npop
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

    X .= clamp.(X, lb, ub)

    for i = 1:npop
        Fringes_fitness[i] = objfun(X[i, :])
    end

    sorted_indices = sortperm(Fringes_fitness)
    Fringes_fitness = Fringes_fitness[sorted_indices]
    X = X[sorted_indices, :]
    best_fringe_fitness = Fringes_fitness[1]
    best_fringe = X[1, :]

    Int_max0 = 1e-20
    cgcurve = zeros(max_iter)
    iter = 1
    X_New = zeros(npop, dim)

    while iter <= max_iter
        cgcurve[iter] = best_fringe_fitness

        sorted_indices = sortperm(Fringes_fitness)
        Fringes_fitness = Fringes_fitness[sorted_indices]
        X = X[sorted_indices, :]

        q = iter / max_iter
        Int_max = Int_max0 * q

        for i = 1:npop
            a = iter^(2 * rand() - 1)
            H = 2 .* rand(dim) .- 1
            z = a ./ H
            r1 = rand()

            if i == 1
                beta = q * cosh(pi / iter)
                A_bright = 2 / (1 + sqrt(abs(1 - beta^2)))
                A = 1:2:npop
                randomIndex = rand(A)
                X_New[i, :] = best_fringe .+ Int_max .* A_bright .* X[i, :] .- r1 .* z .* X[randomIndex, :]
            elseif isodd(i)
                beta = q * cosh(pi / iter)
                A_bright = 2 / (1 + sqrt(abs(1 - beta^2)))
                m = i - 1
                y_bright = Lambda * L * m / d
                Int_bright = Int_max .* cos((pi * d) / (Lambda * L) * y_bright) .^ 2
                s = randperm(npop)[1:2]
                g = 2 * rand() - 1
                Y = X[s[2], :] .- X[s[1], :]
                X_New[i, :] = X[i, :] .- ((1 - g) .* A_bright .* Int_bright .* X[i, :] .+ g .* Y)
            else
                A_dark = Delta * atanh(-(q) + 1)
                m = i - 1
                y_dark = Lambda * L * (m + 0.5) / d
                Int_dark = Int_max .* cos((pi * d) / (Lambda * L) * y_dark) .^ 2
                X_New[i, :] = X[i, :] .- (r1 .* A_dark .* Int_dark .* X[i, :] .- z .* best_fringe)
            end

            X_New[i, :] = clamp.(X_New[i, :], lb, ub)
        end

        X_New_Fitness = zeros(npop)
        for i = 1:npop
            X_New_Fitness[i] = objfun(X_New[i, :])
            if X_New_Fitness[i] < Fringes_fitness[i]
                X[i, :] = X_New[i, :]
                Fringes_fitness[i] = X_New_Fitness[i]
                if Fringes_fitness[i] < best_fringe_fitness
                    best_fringe_fitness = Fringes_fitness[i]
                    best_fringe = X[i, :]
                end
            end
        end

        iter += 1
    end

    # return best_fringe_fitness, best_fringe, cgcurve
    return OptimizationResult(
        best_fringe,
        best_fringe_fitness,
        cgcurve)
end

function YDSE(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return YDSE(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end