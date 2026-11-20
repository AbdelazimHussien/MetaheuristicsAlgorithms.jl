"""
# References:

- Arora, Sankalap, and Satvir Singh. "Butterfly optimization algorithm: a novel approach for global optimization." Soft computing 23 (2019): 715-734.

"""
function BOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return BOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function BOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    p = 0.6
    power_exponent = 0.1
    sensory_modality = 0.01

    Sol = initialization(npop, dim, ub, lb)
    Fitness = zeros(npop)
    for i = 1:npop
        Fitness[i] = objfun(Sol[i, :])
    end

    fmin, I = findmin(Fitness)
    best_pos = Sol[I, :]

    S = copy(Sol)
    Convergence_curve = zeros(max_iter)

    for t = 1:max_iter
        for i = 1:npop
            Fnew = objfun(S[i, :])
            FP = sensory_modality * (Fnew^power_exponent)

            if rand() < p
                dis = rand() * rand() * best_pos - Sol[i, :]  # Eq. (2) in paper
                S[i, :] = Sol[i, :] + dis * FP
            else
                epsilon = rand()
                JK = randperm(npop)
                dis = epsilon * epsilon * Sol[JK[1], :] - Sol[JK[2], :]
                S[i, :] = Sol[i, :] + dis * FP  # Eq. (3) in paper
            end

            Flag4Ub = S[i, :] .> ub
            Flag4Lb = S[i, :] .< lb
            S[i, :] = (S[i, :] .* .~(Flag4Ub .| Flag4Lb)) .+ ub .* Flag4Ub .+ lb .* Flag4Lb

            Fnew = objfun(S[i, :])

            if Fnew <= Fitness[i]
                Sol[i, :] = S[i, :]
                Fitness[i] = Fnew
            end

            if Fnew <= fmin
                best_pos = S[i, :]
                fmin = Fnew
            end
        end

        Convergence_curve[t] = fmin

        sensory_modality = sensory_modality_NEW(sensory_modality, max_iter)
    end

    # return fmin, best_pos, Convergence_curve
    return OptimizationResult(
        best_pos,
        fmin,
        Convergence_curve)
end

function BOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return BOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function sensory_modality_NEW(x, Ngen)
    return x + (0.025 / (x * Ngen))
end

