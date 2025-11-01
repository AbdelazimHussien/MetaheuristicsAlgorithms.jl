"""
# References:

- Hashim, Fatma A., and Abdelazim G. Hussien.
  "Snake Optimizer: A novel meta-heuristic optimization algorithm."
  Knowledge-Based Systems 242 (2022): 108320.
"""
function SO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb) #(npop, max_iter, objfun, dim, lb, ub) #npop, max_iter, lb, ub, dim, Chung_Reynolds #function SO(npop, max_iter, objfun, dim, lb, ub)
    vec_flag = [1, -1]
    Threshold = 0.25
    Thresold2 = 0.6
    C1 = 0.5
    C2 = 0.05
    C3 = 2
    if length(size(lb)) == 0
        lb = fill(lb, dim)
    end
    if length(size(ub)) == 0
        ub = fill(ub, dim)
    end

    X = initialization(npop, dim, ub, lb)
    gbest_t = zeros(max_iter, 1)
    fitness = zeros(npop)
    for i = 1:npop
        fitness[i] = objfun(X[i, :])
    end
    GYbest, gbest = findmin(fitness)
    x_food = X[gbest, :]
    Nm = round(Int, npop / 2)
    Nf = npop - Nm
    Xm = X[1:Nm, :]
    Xnewm = zeros(Nm, dim)
    Xf = X[Nm+1:npop, :]
    Xnewf = zeros(Nf, dim)
    fitness_m = fitness[1:Nm]
    fitness_f = fitness[Nm+1:npop]
    fitnessBest_m, gbest1 = findmin(fitness_m)
    Xbest_m = Xm[gbest1, :]
    fitnessBest_f, gbest2 = findmin(fitness_f)
    Xbest_f = Xf[gbest2, :]
    for t = 1:max_iter
        Temp = exp(-t / max_iter)
        Q = C1 * exp((t - max_iter) / max_iter)
        if Q > 1
            Q = 1
        end

        if Q < Threshold
            for i = 1:Nm
                for j = 1:dim
                    rand_leader_index = rand(1:Nm)
                    X_randm = Xm[rand_leader_index, :]
                    flag_index = rand(1:2)
                    Flag = vec_flag[flag_index]
                    Am = exp(-fitness_m[rand_leader_index] / (fitness_m[i] + eps()))
                    Xnewm[i, j] = X_randm[j] + Flag * C2 * Am * ((ub[j] - lb[j]) * rand() + lb[j])
                end
            end
            for i = 1:Nf
                for j = 1:dim
                    rand_leader_index = rand(1:Nf)
                    X_randf = Xf[rand_leader_index, :]
                    flag_index = rand(1:2)
                    Flag = vec_flag[flag_index]
                    Af = exp(-fitness_f[rand_leader_index] / (fitness_f[i] + eps()))
                    Xnewf[i, j] = X_randf[j] + Flag * C2 * Af * ((ub[j] - lb[j]) * rand() + lb[j])
                end
            end
        else
            if Temp > Thresold2
                for i = 1:Nm
                    flag_index = rand(1:2)
                    Flag = vec_flag[flag_index]
                    for j = 1:dim
                        Xnewm[i, j] = x_food[j] + C3 * Flag * Temp * rand() * (x_food[j] - Xm[i, j])
                    end
                end
                for i = 1:Nf
                    flag_index = rand(1:2)
                    Flag = vec_flag[flag_index]
                    for j = 1:dim
                        Xnewf[i, j] = x_food[j] + Flag * C3 * Temp * rand() * (x_food[j] - Xf[i, j])
                    end
                end
            else
                if rand() > 0.6
                    for i = 1:Nm
                        for j = 1:dim
                            FM = exp(-fitnessBest_f / (fitness_m[i] + eps()))
                            Xnewm[i, j] = Xm[i, j] + C3 * FM * rand() * (Q * Xbest_f[j] - Xm[i, j])
                        end
                    end
                    for i = 1:Nf
                        for j = 1:dim
                            FF = exp(-fitnessBest_m / (fitness_f[i] + eps()))
                            Xnewf[i, j] = Xf[i, j] + C3 * FF * rand() * (Q * Xbest_m[j] - Xf[i, j])
                        end
                    end
                else
                    for i = 1:Nm
                        for j = 1:dim
                            Mm = exp(-fitness_f[i] / (fitness_m[i] + eps()))
                            Xnewm[i, j] = Xm[i, j] + C3 * rand() * Mm * (Q * Xf[i, j] - Xm[i, j])
                        end
                    end
                    for i = 1:Nf
                        for j = 1:dim
                            Mf = exp(-fitness_m[i] / (fitness_f[i] + eps()))
                            Xnewf[i, j] = Xf[i, j] + C3 * rand() * Mf * (Q * Xm[i, j] - Xf[i, j])
                        end
                    end
                    flag_index = rand(1:2)
                    egg = vec_flag[flag_index]
                    if egg == 1
                        GYworst, gworst = findmax(fitness_m)
                        Xnewm[gworst, :] = lb .+ rand(dim) .* (ub .- lb)
                        GYworst, gworst = findmax(fitness_f)
                        Xnewf[gworst, :] = lb .+ rand(dim) .* (ub .- lb)
                    end
                end
            end
        end

        for j = 1:Nm
            Xnewm[j, :] = max.(min.(Xnewm[j, :], ub), lb)
            y = objfun(Xnewm[j, :])
            if y < fitness_m[j]
                fitness_m[j] = y
                Xm[j, :] = Xnewm[j, :]
            end
        end

        Ybest1, gbest1 = findmin(fitness_m)

        for j = 1:Nf
            Xnewf[j, :] = max.(min.(Xnewf[j, :], ub), lb)
            y = objfun(Xnewf[j, :])
            if y < fitness_f[j]
                fitness_f[j] = y
                Xf[j, :] = Xnewf[j, :]
            end
        end

        Ybest2, gbest2 = findmin(fitness_f)

        if Ybest1 < fitnessBest_m
            Xbest_m = Xm[gbest1, :]
            fitnessBest_m = Ybest1
        end
        if Ybest2 < fitnessBest_f
            Xbest_f = Xf[gbest2, :]
            fitnessBest_f = Ybest2
        end
        if Ybest1 < Ybest2
            gbest_t[t] = minimum(Ybest1)
        else
            gbest_t[t] = minimum(Ybest2)
        end
        if fitnessBest_m < fitnessBest_f
            GYbest = fitnessBest_m
            x_food = Xbest_m
        else
            GYbest = fitnessBest_f
            x_food = Xbest_f
        end
    end
    fval = GYbest
    # return fval, x_food, gbest_t
    return OptimizationResult(
        x_food,
        fval,
        gbest_t)
end

function SO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end