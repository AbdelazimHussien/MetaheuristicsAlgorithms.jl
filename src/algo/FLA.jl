"""
# References:

- Hashim, Fatma A., Reham R. Mostafa, Abdelazim G. Hussien, Seyedali Mirjalili, and Karam M. Sallam.
  "Fick's Law Algorithm: A physical law-based algorithm for numerical optimization."
  Knowledge-Based Systems 260 (2023): 110146.
"""
function FLA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return FLA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function FLA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, NoMolecules::Integer, max_iter::Integer)
    dim = length(lb)
    C1 = 0.5
    C2 = 2
    c3 = 0.1
    c4 = 0.2
    c5 = 2
    D = 0.01
    # X = lb .+ rand(NoMolecules, dim) .* (ub .- lb)
    X = initialization(NoMolecules, dim, ub, lb)
    X1new = zeros(NoMolecules, dim)
    X2new = zeros(NoMolecules, dim)
    CNVG = zeros(max_iter, 1)
    FS = zeros(NoMolecules)
    TF = zeros(max_iter)
    for i = 1:NoMolecules
        FS[i] = objfun(X[i, :])
    end

    BestF, IndexBestF = findmin(FS)
    Xss = X[IndexBestF, :]

    n1 = round(Int, NoMolecules / 2)
    n2 = NoMolecules - n1

    X1 = X[1:n1, :]
    X2 = X[n1+1:NoMolecules, :]

    FS1 = zeros(n1)
    FS2 = zeros(n2)

    for i = 1:n1
        FS1[i] = objfun(X1[i, :])
    end

    for i = 1:n2
        FS2[i] = objfun(X2[i, :])
    end

    FSeo1, IndexFSeo1 = findmin(FS1)
    FSeo2, IndexFSeo2 = findmin(FS2)

    Xeo1 = X1[IndexFSeo1, :]
    Xeo2 = X2[IndexFSeo2, :]

    vec_flag = [1, -1]

    if FSeo1 < FSeo2
        FSss = FSeo1
        YSol = Xeo1
    else
        FSss = FSeo2
        YSol = Xeo2
    end

    for t = 1:max_iter
        TF[t] = sinh(t / max_iter)^C1

        if TF[t] < 0.9
            DOF = exp(-(C2 * TF[t] - rand()))^C2
            TDO = c5 * TF[t] - rand()

            if TDO < rand()
                M1N = c3 * n1
                M2N = c4 * n1
                NT12 = round(Int, (M2N - M1N) * rand() + M1N)

                for u = 1:NT12
                    flag_index = rand(1:2)
                    DFg = vec_flag[flag_index]
                    Xm2 = mean(X2)
                    Xm1 = mean(X1)
                    J = -D * (Xm2 - Xm1) / (norm(Xeo2 - X1[u, :]) + eps())
                    X1new[u, :] = Xeo2 .+ rand(dim) .* DFg .* DOF .* (J .* Xeo2 - X1[u, :])
                end

                for u = NT12+1:n1
                    for tt = 1:dim
                        p = rand()
                        if p < 0.8
                            X1new[u, tt] = Xeo1[tt]
                        elseif p < 0.9
                            r3 = rand()
                            X1new[u, tt] = X1[u, tt] + DOF * ((ub - lb) * r3 + lb)
                        else
                            X1new[u, tt] = X1[u, tt]
                        end
                    end
                end

                for u = 1:n2
                    r4 = rand()
                    X2new[u, :] = Xeo2 .+ DOF .* ((ub - lb) .* r4 .+ lb)
                end
            else
                M1N = 0.1 * n2
                M2N = 0.2 * n2
                Ntransfer = round(Int, (M2N - M1N) * rand() + M1N)

                for u = 1:Ntransfer
                    flag_index = rand(1:2)
                    DFg = vec_flag[flag_index]
                    R1 = rand(1:n1)
                    Xm1 = mean(X1, dims=1)
                    Xm2 = mean(X2, dims=1)
                    J = (-D .* (Xm1 - Xm2)) ./ norm((Xeo1 .- X2[u, :]) .+ eps())
                    X2new[u, :] = Xeo1 .+ DFg .* DOF .* rand(dim) .* (transpose(J) .* Xeo1 .- X2[u, :])
                end

                for u = Ntransfer+1:n2
                    for tt = 1:dim
                        p = rand()
                        if p < 0.8
                            X2new[u, tt] = Xeo2[tt]
                        elseif p < 0.9
                            r3 = rand()
                            X2new[u, tt] = X2[u, tt] + DOF * ((ub - lb) * r3 + lb)
                        else
                            X2new[u, tt] = X2[u, tt]
                        end
                    end
                end

                for u = 1:n1
                    r4 = rand()
                    X1new[u, :] = Xeo1 .+ DOF .* ((ub .- lb) .* r4 + lb)
                end
            end
        else
            if TF[t] <= 1
                for u = 1:n1
                    flag_index = rand(1:2)
                    DFg = vec_flag[flag_index]
                    Xm1 = mean(X1)
                    Xmeo1 = Xeo1
                    J = -D .* (Xmeo1 .- Xm1) / (norm((Xeo1 - X1[u, :])) + eps())
                    DRF = exp.(-J / TF[t])
                    MS = exp(-FSeo1 / (FS1[u] + eps()))
                    R1 = rand(dim)
                    Qeo = DFg * DRF .* R1
                    X1new[u, :] = Xeo1 + Qeo .* X1[u, :] + Qeo .* (MS * Xeo1 - X1[u, :])
                end

                for u = 1:n2
                    flag_index = rand(1:2)
                    DFg = vec_flag[flag_index]
                    Xm2 = mean(X2)
                    Xmeo2 = Xeo2
                    J = -D .* (Xmeo2 .- Xm2) ./ (norm((Xeo2 .- X2[u, :])) + eps())
                    DRF = exp.(-J / TF[t])
                    MS = exp(-FSeo2 / (FS2[u] + eps()))
                    R1 = rand(dim)
                    Qeo = DFg * DRF .* R1
                    X2new[u, :] = Xeo2 + Qeo .* X2[u, :] + Qeo .* (MS * Xeo1 - X2[u, :])
                end
            else
                R1 = rand(1, dim)
                for u = 1:n1
                    flag_index = rand(1:2)
                    DFg = vec_flag[flag_index]
                    Xm1 = mean(X1)
                    Xm = mean(X)
                    J = -D .* (Xm .- Xm1) / (norm(Xss - X1[u, :]) + eps())
                    DRF = exp(-J / TF[t])
                    MS = exp(-FSss / (FS1[u] + eps()))
                    R1 = rand(dim)
                    Qg = DFg * DRF .* R1
                    X1new[u, :] = Xss + Qg .* X1[u, :] + Qg .* (MS * Xss - X1[u, :])
                end

                for u = 1:n2
                    Xm1 = mean(X1)
                    Xm = mean(X)
                    J = -D .* (Xm1 - Xm) / (norm(Xss - X2[u, :]) + eps())
                    DRF = exp(-J / TF[t])
                    MS = exp(-FSss / (FS2[u] + eps()))
                    flag_index = rand(1:2)
                    DFg = vec_flag[flag_index]
                    Qg = DFg * DRF .* R1
                    X2new[u, :] = Xss + Qg .* X2[u, :] + Qg .* (MS * Xss - X2[u, :])
                end
            end
        end
        for j = 1:n1
            FU = X1new[j, :] .> ub
            FL = X1new[j, :] .< lb
            X1new[j, :] = max.(min.(X2new[j, :], ub), lb)
            v = objfun(X1new[j, :])
            if v < FS1[j]
                FS1[j] = v
                X1[j, :] = X1new[j, :]
            end
        end

        for j = 1:n2
            FU = X2new[j, :] .> ub
            FL = X2new[j, :] .< lb
            X2new[j, :] = max.(min.(X2new[j, :], ub), lb)
            v = objfun(X2new[j, :])
            if v < FS2[j]
                FS2[j] = v
                X2[j, :] = X2new[j, :]
            end
        end

        FSeo1, IndexFSeo1 = findmin(FS1)
        FSeo2, IndexFSeo2 = findmin(FS2)

        Xeo1 = X1[IndexFSeo1, :]
        Xeo2 = X2[IndexFSeo2, :]

        if FSeo1 < FSeo2
            FSss = FSeo1
            YSol = Xeo1
        else
            FSss = FSeo2
            YSol = Xeo2
        end

        CNVG[t] = FSss

        if FSss < BestF
            BestF = FSss
            Xss = YSol
        end
    end
    # return BestF, Xss, CNVG
    return OptimizationResult(
        Xss,
        BestF,
        CNVG)
end

function FLA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return FLA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end