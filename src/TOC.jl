# Tornado Optimizer with Coriolis Force (TOC) in Julia

function TOC(n, max_it, lb, ub, dim, fobj, nto=4, nt=3)
    ccurve = zeros(max_it)

    # Initialization
    y = initialization(n, dim, ub, lb)
    fit = [fobj(y[i, :]) for i in 1:axes(y, 1)]
    sorted_idx = sortperm(fit)

    To = nto - nt
    nw = n - nto

    Tornadoposition = y[sorted_idx[1:To], :]
    TornadoCost = fit[sorted_idx[1:To]]

    Thunderstormsposition = y[sorted_idx[2:nto], :]
    ThunderstormsCost = fit[sorted_idx[2:nto]]

    bThunderstormsCost = copy(ThunderstormsCost)
    gThunderstormsCost = zeros(nto - 1)
    ind = argmin(ThunderstormsCost)

    bThunderstormsposition = copy(Thunderstormsposition)
    gThunderstormsCost = Thunderstormsposition[ind, :]

    Windstormsposition = y[sorted_idx[nto+1:end], :]
    WindstormsCost = fit[sorted_idx[nto+1:end]]

    gWindstormsposition = zeros(nw)
    bWindstormsCost = copy(WindstormsCost)
    ind = argmin(WindstormsCost)
    bWindstormsposition = copy(Windstormsposition)
    gWindstormsposition = Windstormsposition[ind, :]

    vel_storm = 0.1 .* Windstormsposition

    nwindstorms = sort(randperm(nw)[1:nto])
    nWT = diff([0; nwindstorms])
    push!(nWT, nw - sum(nWT))

    nWT1 = nWT[1]
    nWH = nWT[2:end]

    b_r = 100000
    fdelta = [-1, 1]

    chi = 4.10
    eta = 2 / abs(2 - chi - sqrt(chi^2 - 4 * chi))

    t = 1
    println("================  Tornado Optimizer with Coriolis force (TOC) ================ ")

    while t <= max_it
        nu = (0.1 * exp(-0.1 * (t / max_it)^0.1))^16
        mu = 0.5 + rand() / 2
        ay = (max_it - (t^2 / max_it)) / max_it

        Rl = 2 / (1 + exp((-t + max_it / 2) / 2))
        Rr = -2 / (1 + exp((-t + max_it / 2) / 2))

        # TODO: Continue porting core update loops (windstorms to tornado, exploitation, etc.)

        println("Iteration: $t   minTornadoCost= $(minimum(TornadoCost))")
        ccurve[t] = minimum(TornadoCost)
        t += 1
    end

    return TornadoCost, Tornadoposition, ccurve
end

