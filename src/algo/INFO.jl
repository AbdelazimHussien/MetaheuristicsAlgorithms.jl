"""
# References:

- Ahmadianfar, Iman, Ali Asghar Heidari, Saeed Noshadian, Huiling Chen, and Amir H. Gandomi.
  "INFO: An efficient optimization algorithm based on weighted mean of vectors."
  Expert Systems with Applications 195 (2022): 116516.
"""
function INFO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return INFO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function INFO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)::OptimizationResult
    dim = length(lb)
    # Initialization
    Cost = zeros(npop)
    M = zeros(npop)

    X = initialization(npop, dim, ub, lb)

    for i = 1:npop
        Cost[i] = objfun(X[i, :])
        M[i] = Cost[i]
    end

    sorted_inds = sortperm(Cost)
    Best_X = X[sorted_inds[1], :]
    Best_Cost = Cost[sorted_inds[1]]

    Worst_Cost = Cost[sorted_inds[end]]
    Worst_X = X[sorted_inds[end], :]

    I = rand(2:5)
    Better_X = X[sorted_inds[I], :]
    Better_Cost = Cost[sorted_inds[I]]

    # Main Loop of INFO
    Convergence_curve = zeros(max_iter)

    for it = 1:max_iter
        alpha = 2 * exp(-4 * (it / max_iter))  # Eq. (5.1) & Eq. (9.1)

        M_Best = Best_Cost
        M_Better = Better_Cost
        M_Worst = Worst_Cost

        for i = 1:npop
            # Updating rule stage
            del = 2 * rand() * alpha - alpha  # Eq. (5)
            sigm = 2 * rand() * alpha - alpha  # Eq. (9)

            # Select three random solutions
            A1 = randperm(npop)
            A1 = A1[A1.!=i]
            a, b, c = A1[1:3]

            e = 1e-25
            epsi = e * rand()

            omg = maximum([M[a], M[b], M[c]])
            MM = [M[a] - M[b], M[a] - M[c], M[b] - M[c]]

            W = cos.(MM .+ π) .* exp.(-MM ./ omg)  # Eqs. (4.2), (4.3), (4.4)
            Wt = sum(W)

            WM1 = del * (W[1] * (X[a, :] - X[b, :]) + W[2] * (X[a, :] - X[c, :]) +
                         W[3] * (X[b, :] - X[c, :])) / (Wt + 1) .+ epsi

            omg = maximum([M_Best, M_Better, M_Worst])
            MM = [M_Best - M_Better, M_Best - M_Worst, M_Better - M_Worst]

            W = cos.(MM .+ π) .* exp.(-MM ./ omg)  # Eqs. (4.7), (4.8), (4.9)
            Wt = sum(W)

            WM2 = del * (W[1] * (Best_X - Better_X) + W[2] * (Best_X - Worst_X) +
                         W[3] * (Better_X - Worst_X)) / (Wt + 1) .+ epsi

            # Determine MeanRule
            r = rand(Uniform(0.1, 0.5))
            MeanRule = r * WM1 + (1 - r) * WM2  # Eq. (4)

            if rand() < 0.5
                z1 = X[i, :] + sigm * (rand() * MeanRule) + randn() * (Best_X - X[a, :]) / (M_Best - M[a] + 1)
                z2 = Best_X + sigm * (rand() * MeanRule) + randn() * (X[a, :] - X[b, :]) / (M[a] - M[b] + 1)
            else  # Eq. (8)
                z1 = X[a, :] + sigm * (rand() * MeanRule) + randn() * (X[b, :] - X[c, :]) / (M[b] - M[c] + 1)
                z2 = Better_X + sigm * (rand() * MeanRule) + randn() * (X[a, :] - X[b, :]) / (M[a] - M[b] + 1)
            end

            # Vector combining stage
            u = zeros(dim)
            for j = 1:dim
                mu = 0.05 * randn()
                if rand() < 0.5
                    if rand() < 0.5
                        u[j] = z1[j] + mu * abs(z1[j] - z2[j])  # Eq. (10.1)
                    else
                        u[j] = z2[j] + mu * abs(z1[j] - z2[j])  # Eq. (10.2)
                    end
                else
                    u[j] = X[i, j]  # Eq. (10.3)
                end
            end

            # Local search stage
            if rand() < 0.5
                L = rand() < 0.5
                v1 = (1 - L) * 2 * rand() + L
                v2 = rand() * L + (1 - L)
                Xavg = (X[a, :] + X[b, :] + X[c, :]) / 3  # Eq. (11.4)
                phi = rand()
                Xrnd = phi * Xavg + (1 - phi) * (phi * Better_X + (1 - phi) * Best_X)  # Eq. (11.3)
                Randn = L * randn(dim) + (1 - L) * randn(dim)
                if rand() < 0.5
                    u = Best_X + Randn .* (MeanRule + randn(dim) .* (Best_X - X[a, :]))  # Eq. (11.1)
                else
                    u = Xrnd + Randn .* (MeanRule + randn(dim) .* (v1 * Best_X - v2 * Xrnd))  # Eq. (11.2)
                end
            end

            # Check if new solution is outside the search space and bring it back
            # New_X = BC(u, lb, ub)
            New_X = max.(min.(u, ub), lb)
            New_Cost = objfun(New_X)

            if New_Cost < Cost[i]
                X[i, :] = New_X
                Cost[i] = New_Cost
                M[i] = Cost[i]
                if Cost[i] < Best_Cost
                    Best_X = X[i, :]
                    Best_Cost = Cost[i]
                end
            end
        end

        # Determine the worst and better solutions
        sorted_inds = sortperm(Cost)
        Worst_X = X[sorted_inds[end], :]
        Worst_Cost = Cost[sorted_inds[end]]
        I = rand(2:5)
        Better_X = X[sorted_inds[I], :]
        Better_Cost = Cost[sorted_inds[I]]

        # Update Convergence_curve
        Convergence_curve[it] = Best_Cost

        # Show Iteration Information
        # println("Iteration $it: Best Cost = $Best_Cost")
    end

    # return Best_Cost, Best_X, Convergence_curve
    return OptimizationResult(
        Best_X,
        Best_Cost,
        Convergence_curve)
end

function INFO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return INFO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end