"""
# References:

- Braik, Malik Shehadeh. "Chameleon Swarm Algorithm: A bio-inspired optimizer for solving engineering design problems." Expert Systems with Applications 174 (2021): 114685.

"""
function ChameleonSA(searchAgents, iteMax, lb, ub, dim, fobj)
    if length(ub) == 1
        ub = fill(ub, dim)
        lb = fill(lb, dim)
    end

    cg_curve = zeros(Float64, iteMax)

    chameleonPositions = initialization(searchAgents, dim, ub, lb)

    fit = [fobj(chameleonPositions[i, :]) for i in 1:searchAgents]

    fitness = copy(fit)
    fmin0, index = findmin(fit)

    chameleonBestPosition = copy(chameleonPositions)
    gPosition = chameleonPositions[index, :]
    v = 0.1 .* chameleonBestPosition
    v0 = 0.0 .* v

    rho, p1, p2, c1, c2, gamma, alpha, beta = 1.0, 2.0, 2.0, 2.0, 1.80, 2.0, 4.0, 3.0

    for t in 1:iteMax
        a = 2590 * (1 - exp(-log(t)))
        omega = (1 - (t / iteMax))^(rho * sqrt(t / iteMax))
        p1 = 2 * exp(-2 * (t / iteMax)^2)
        p2 = 2 / (1 + exp((-t + iteMax / 2) / 100))
        mu = gamma * exp(-(alpha * t / iteMax)^beta)

        ch = ceil.(Int, searchAgents * rand(searchAgents))

        for i in 1:searchAgents
            if rand() >= 0.1
                chameleonPositions[i, :] .= chameleonPositions[i, :] + 
                    p1 .* (chameleonBestPosition[ch[i], :] - chameleonPositions[i, :]) .* rand() +
                    p2 .* (gPosition - chameleonPositions[i, :]) .* rand()
            else
                for j in 1:dim
                    chameleonPositions[i, j] = gPosition[j] + mu * ((ub[j] - lb[j]) * rand() + lb[j]) * sign(rand() - 0.5)
                end
            end
        end

        for i in 1:searchAgents
            v[i, :] .= omega .* v[i, :] + 
                p1 .* (chameleonBestPosition[i, :] - chameleonPositions[i, :]) .* rand() +
                p2 .* (gPosition - chameleonPositions[i, :]) .* rand()

            chameleonPositions[i, :] .= chameleonPositions[i, :] + (v[i, :].^2 - v0[i, :].^2) / (2 * a)
        end
        v0 .= v

        for i in 1:searchAgents
            for j in 1:dim
                if chameleonPositions[i, j] < lb[j]
                    chameleonPositions[i, j] = lb[j]
                elseif chameleonPositions[i, j] > ub[j]
                    chameleonPositions[i, j] = ub[j]
                end
            end
        end

        for i in 1:searchAgents
            chameleonPositions[i, :] = clamp.(chameleonPositions[i, :], lb, ub)

            fit[i] = fobj(chameleonPositions[i, :])
            if fit[i] < fitness[i]
                chameleonBestPosition[i, :] = chameleonPositions[i, :]
                fitness[i] = fit[i]
            end
        end

        fmin, index = findmin(fitness)

        if fmin < fmin0
            gPosition = chameleonBestPosition[index, :]
            fmin0 = fmin
        end

        cg_curve[t] = fmin0
    end

    return fmin0, gPosition, cg_curve
end