"""
Dhiman, Gaurav, and Vijay Kumar. 
"Seagull optimization algorithm: Theory and its applications for large-scale industrial engineering problems." 
Knowledge-based systems 165 (2019): 169-196.
"""
function SOA(npop::Int, max_iter::Int, lb, ub, dim::Int, objfun)
    position = zeros(dim)
    score = Inf
    positions = initialization(npop, dim, ub, lb)
    convergence = zeros(max_iter)

    l = 0

    while l < max_iter
        for i in axes(positions, 1)
            positions[i, :] = clamp.(positions[i, :], lb, ub)

            fitness = objfun(positions[i, :])

            if fitness < score
                score = fitness
                position = copy(positions[i, :])
            end
        end

        Fc = 2 - l * (2 / max_iter)

        for i in axes(positions, 1)
            for j in axes(positions, 2)
                r1 = rand()
                r2 = rand()

                A1 = 2 * Fc * r1 - Fc
                C1 = 2 * r2
                b = 1
                ll = (Fc - 1) * rand() + 1

                D_alpha = Fc * positions[i, j] + A1 * (position[j] - positions[i, j])
                X1 = D_alpha * exp(b * ll) * cos(ll * 2 * π) + position[j]
                positions[i, j] = X1
            end
        end

        l += 1
        convergence[l] = score
        println("convergence[$l] ", score)
    end

    # return score, position, convergence
    return OptimizationResult(
        position,
        score,
        convergence)
end
