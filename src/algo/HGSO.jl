"""
# References:

- Hashim, F.A., Houssein, E.H., Mabrouk, M.S., Al-Atabany, W. and Mirjalili, S., 2019.
  Henry gas solubility optimization: A novel physics-based algorithm.
  Future Generation Computer Systems, 101, pp.646-667.
"""
function HGSO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return HGSO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end
function HGSO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, var_n_gases::Integer, max_iter::Integer)
    dim = length(lb)
    var_n_types = 5

    l1 = 5e-3
    l2 = 100
    l3 = 1e-2

    alpha = 1
    beta = 1

    M1 = 0.1
    M2 = 0.2

    K = l1 * rand(var_n_types)
    P = l2 * rand(var_n_gases)
    C = l3 * rand(var_n_types)

    # X = lb .+ rand(var_n_gases, dim) .* (ub - lb)
    X = initialization(var_n_gases, dim, ub, lb)

    Group = Create_Groups(var_n_gases, var_n_types, X)

    best_fit = zeros(var_n_types)
    best_pos = Vector{Any}(undef, var_n_types)

    for i = 1:var_n_types
        Group[i], best_fit[i], best_pos[i] = Evaluate(objfun, var_n_types, var_n_gases, Group[i], 0, 1)
    end

    var_Gbest, var_gbest = findmin(best_fit)
    vec_Xbest = best_pos[var_gbest]

    vec_Gbest_iter = zeros(max_iter)

    for var_iter = 1:max_iter
        S = update_variables(var_iter, max_iter, K, P, C, var_n_types, var_n_gases)
        Groupnew = update_positions(Group, best_pos, vec_Xbest, S, var_n_gases, var_n_types, var_Gbest, alpha, beta, dim)
        Groupnew = fun_checkpoisions(dim, Groupnew, var_n_gases, var_n_types, lb, ub)

        for i = 1:var_n_types
            Group[i], best_fit[i], best_pos[i] = Evaluate(objfun, var_n_types, var_n_gases, Group[i], Groupnew[i], 0)
            Group[i] = worst_agents(Group[i], M1, M2, dim, ub, lb, var_n_gases, var_n_types)
        end

        var_Ybest, var_index = findmin(best_fit)
        vec_Gbest_iter[var_iter] = var_Ybest

        if var_Ybest < var_Gbest
            var_Gbest = var_Ybest
            vec_Xbest = best_pos[var_index]
        end
    end

    # return var_Gbest, vec_Xbest, vec_Gbest_iter
    return OptimizationResult(
        vec_Xbest,
        var_Gbest,
        vec_Gbest_iter)
end

function Create_Groups(var_n_gases, var_n_types, X)
    N = div(var_n_gases, var_n_types)
    Group = Vector{Dict{Symbol,Any}}(undef, var_n_types)

    i = 1
    for j = 1:var_n_types
        Group[j] = Dict(:Position => X[i:i+N-1, :])

        i = j * N + 1
        if i + N > var_n_gases
            i = j * N
        end
    end

    return Group
end

function Evaluate(objfun, var_n_types, var_n_gases, X, Xnew, init_flag)
    N = div(var_n_gases, var_n_types)

    if !haskey(X, :fitness)
        X[:fitness] = zeros(N)
    end

    if init_flag == 1
        for j = 1:N
            X[:fitness][j] = objfun(X[:Position][j, :])
        end
    else
        for j = 1:N
            temp_fit = objfun(Xnew[:Position][j, :])
            if temp_fit < X[:fitness][j]
                X[:fitness][j] = temp_fit
                X[:Position][j, :] = Xnew[:Position][j, :]
            end
        end
    end

    best_fit, index_best = findmin(X[:fitness])
    best_pos = X[:Position][index_best, :]

    # return X, best_fit, best_pos
    return OptimizationResult(
        X,
        best_fit,
        best_pos)
end

function HGSO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return HGSO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function update_variables(var_iter, max_iter, K, P, C, var_n_types, var_n_gases)
    T = exp(-var_iter / max_iter)
    T0 = 298.15
    N = div(var_n_gases, var_n_types)
    S = zeros(var_n_gases, size(P, 2))

    i = 1
    for j = 1:var_n_types
        K[j] = K[j] * exp(-C[j] * (1 / T - 1 / T0))

        S[i:i+N-1, :] .= P[i:i+N-1, :] * K[j]

        i = j * N + 1
        if i + N > var_n_gases
            i = j * N
        end
    end

    return S
end

function update_positions(Group, best_pos, vec_Xbest, S, var_n_gases, var_n_types, var_Gbest, alpha, beta, var_nvars)
    vec_flag = [1, -1]

    for i = 1:var_n_types
        for j = 1:div(var_n_gases, var_n_types)
            gama = beta * exp(-(var_Gbest + 0.05) / (Group[i][:fitness][j] + 0.05))

            flag_index = Int(floor(2 * rand() + 1))
            var_flag = vec_flag[flag_index]

            for k = 1:var_nvars
                Group[i][:Position][j, k] += var_flag * rand() * gama * (best_pos[i][k] - Group[i][:Position][j, k]) + rand() * alpha * var_flag * (S[i] * vec_Xbest[k] - Group[i][:Position][j, k])
            end
        end
    end

    return Group
end

function fun_checkpoisions(dim, Group, var_n_gases, var_n_types, lb, ub)
    lb = ones(dim) * lb
    ub = ones(dim) * ub

    for j = 1:var_n_types
        for i = 1:div(var_n_gases, var_n_types)
            pos = Group[j][:Position][i, :]

            isBelow1 = pos .< lb
            isAboveMax = pos .> ub

            if any(isBelow1)
                Group[j][:Position][i, :] .= lb
            elseif any(isAboveMax)
                Group[j][:Position][i, :] .= ub
            end
        end
    end

    return Group
end

function worst_agents(X, M1, M2, dim, G_max, G_min, var_n_gases, var_n_types)
    X_index = sortperm(X[:fitness], rev=true)
    X_sort = X[:fitness][X_index]

    M1N = M1 * var_n_gases / var_n_types
    M2N = M2 * var_n_gases / var_n_types

    Nw = round(Int, (M2N - M1N) * rand() + M1N)

    for k = 1:Nw
        idx = X_index[k]
        X[:Position][idx, :] .= G_min .+ rand(dim) .* (G_max .- G_min)
    end

    return X
end