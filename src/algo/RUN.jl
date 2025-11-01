"""
# References:

- Ahmadianfar, Iman, Ali Asghar Heidari, Amir H. Gandomi, Xuefeng Chu, and Huiling Chen.
  "RUN beyond the metaphor: An efficient optimization algorithm based on Runge Kutta method."
  Expert Systems with Applications 181 (2021): 115079.
"""
function RUN(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return RUN(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function RUN(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    Cost = zeros(npop)
    X = initialization(npop, dim, ub, lb)
    Xnew2 = zeros(dim)

    Convergence_curve = zeros(max_iter)

    for i in 1:npop
        Cost[i] = objfun(X[i, :])
    end

    Best_Cost, ind = findmin(Cost)
    Best_X = X[ind, :]

    Convergence_curve[1] = Best_Cost


    it = 1
    while it < max_iter
        it += 1
        f = 20 * exp(-(12 * (it / max_iter)))
        Xavg = mean(X, dims=1)'
        SF = 2 * (0.5 .- rand(npop)) .* f

        for i in 1:npop
            _, ind_l = findmin(Cost)
            lBest = X[ind_l, :]

            A, B, C = RndX(npop, i)
            _, ind1 = findmin(Cost[[A, B, C]])

            gama = rand() * (X[i, :] .- rand(dim) .* (ub .- lb)) .* exp(-4 * it / max_iter)
            Stp = rand(dim) .* ((Best_X .- rand() .* Xavg) .+ gama)
            DelX = 2 * rand(dim) .* abs.(Stp)

            if Cost[i] < Cost[ind1]
                Xb = X[i, :]
                Xw = X[ind1, :]
            else
                Xb = X[ind1, :]
                Xw = X[i, :]
            end

            SM = RungeKutta(Xb, Xw, DelX)

            L = rand(dim) .< 0.5
            Xc = L .* X[i, :] .+ (1 .- L) .* X[A, :]
            Xm = L .* Best_X .+ (1 .- L) .* lBest

            vec = [1, -1]
            r = [1, -1][Int.(floor.(2 * rand(dim) .+ 1))]

            g = 2 * rand(1)
            mu = 0.5 .+ 0.1 * randn(dim)



            if rand() < 0.5
                Xnew = (Xc .+ r .* SF[i] .* g .* Xc) .+ SF[i] .* SM .+ mu .* (Xm .- Xc)
            else
                Xnew = (Xm .+ r .* SF[i] .* g .* Xm) .+ SF[i] .* SM .+ mu .* (X[A, :] .- X[B, :])
            end

            FU = Xnew .> ub
            FL = Xnew .< lb
            Xnew = (Xnew .* .! (FU .| FL)) .+ ub .* FU .+ lb .* FL

            CostNew = objfun(Xnew[:])

            if CostNew < Cost[i]
                X[i, :] = Xnew
                Cost[i] = CostNew
            end

            if rand() < 0.5
                EXP = exp(-5 * rand() * it / max_iter)
                r = floor(Unifrnd(-1, 2, 1, 1)[1])

                u = 2 * rand(1, dim)
                w = Unifrnd(0, 2, 1, dim) .* EXP

                A, B, C = RndX(npop, i)
                Xavg = (X[A, :] .+ X[B, :] .+ X[C, :]) / 3

                beta = rand(1, dim)
                Xnew1 = beta .* Best_X .+ (1 .- beta) .* Xavg

                for j in 1:dim
                    if w[j] < 1
                        Xnew2[j] = Xnew1[j] .+ r * w[j] * abs((Xnew1[j] - Xavg[j]) + randn())
                    else
                        Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[j] * abs((u[j] .* Xnew1[j] - Xavg[j]) + randn())
                    end
                end

                FU = Xnew2 .> ub
                FL = Xnew2 .< lb
                Xnew2 = (Xnew2 .* .! (FU .| FL)) .+ ub .* FU .+ lb .* FL
                CostNew = objfun(Xnew2)


                if CostNew < Cost[i]
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                else
                    if rand() < w[Int(rand(1:dim))]
                        SM = RungeKutta(X[i, :], Xnew2, DelX)[:, 1]

                        Xnew = (Xnew2 .- rand() .* Xnew2) .+ SF[i] .* (SM .+ (2 * rand(dim) .* Best_X .- Xnew2))  # (Eq. 20)

                        FU = Xnew .> ub
                        FL = Xnew .< lb
                        Xnew = (Xnew .* .! (FU .| FL)) .+ ub .* FU .+ lb .* FL
                        CostNew = objfun(Xnew)

                        if CostNew < Cost[i]
                            X[i, :] = Xnew
                            Cost[i] = CostNew
                        end
                    end
                end
            end

            if Cost[i] < Best_Cost
                Best_X = X[i, :]
                Best_Cost = Cost[i]
            end
        end

        Convergence_curve[it] = Best_Cost
        # println("it : ", it, ", Best Cost = ", Convergence_curve[it])
    end
    # return Best_Cost, Best_X, Convergence_curve
    return OptimizationResult(
        Best_X,
        Best_Cost,
        Convergence_curve)
end

function RUN(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return RUN(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function Unifrnd(a, b, c, dim)
    a2 = a / 2
    b2 = b / 2
    mu = a2 + b2
    sig = b2 - a2
    return mu .+ sig .* (2 * rand(c, dim) .- 1)
end

function RndX(npop, i)
    Qi = randperm(npop)
    filter!(x -> x != i, Qi)
    return Qi[1], Qi[2], Qi[3]
end

function RungeKutta(XB, XW, DelX)
    dim = size(XB, 2)
    C = rand(Int) + 1
    r1 = rand(dim)
    r2 = rand(dim)

    K1 = 0.5 * (rand() * XW .- C .* XB)
    K2 = 0.5 * (rand() * (XW .+ r2 .* K1 .* DelX / 2) .- (C .* XB .+ r1 .* K1 .* DelX / 2))
    K3 = 0.5 * (rand() * (XW .+ r2 .* K2 .* DelX / 2) .- (C .* XB .+ r1 .* K2 .* DelX / 2))
    K4 = 0.5 * (rand() * (XW .+ r2 .* K3 .* DelX) .- (C .* XB .+ r1 .* K3 .* DelX))

    XRK = K1 .+ 2 .* K2 .+ 2 .* K3 .+ K4
    SM = (1 / 6) * XRK

    return SM
end