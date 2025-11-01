
function Interpolation(xi, xj, xk, fi, fj, fk, l, u)
    a = (xj^2 - xk^2) * fi + (xk^2 - xi^2) * fj + (xi^2 - xj^2) * fk
    b = 2 * ((xj - xk) * fi + (xk - xi) * fj + (xi - xj) * fk)

    L_xmin = a / (b + eps())
    if isnan(L_xmin) || isinf(L_xmin) || L_xmin > u || L_xmin < l
        L_xmin = rand() * (u - l) + l
    end
    return L_xmin
end

function GQI(a, b, c, fa, fb, fc, low, up)
    fabc = [fa, fb, fc]
    ind = sortperm(fabc)
    fijk = fabc[ind]

    fi, fj, fk = fijk
    dim = length(a)
    ai, bi, ci = ind

    L = zeros(dim)

    for i = 1:dim
        x = [a[i], b[i], c[i]]
        xi, xj, xk = x[ai], x[bi], x[ci]

        if (xk >= xi && xi >= xj) || (xj >= xi && xi >= xk)
            L[i] = Interpolation(xi, xj, xk, fi, fj, fk, low[i], up[i])
        elseif (xk >= xj && xj >= xi)
            I = Interpolation(xi, xj, xk, fi, fj, fk, low[i], up[i])
            if I < xj
                L[i] = I
            else
                L[i] = Interpolation(xi, xj, 3 * xi - 2 * xj, fi, fj, fk, low[i], up[i])
            end
        elseif (xi >= xj && xj >= xk)
            I = Interpolation(xi, xj, xk, fi, fj, fk, low[i], up[i])
            if I > xj
                L[i] = I
            else
                L[i] = Interpolation(xi, xj, 3 * xi - 2 * xj, fi, fj, fk, low[i], up[i])
            end
        elseif (xj >= xk && xk >= xi)
            L[i] = Interpolation(xi, 2 * xi - xk, xk, fi, fj, fk, low[i], up[i])
        elseif (xi >= xk && xk >= xj)
            L[i] = Interpolation(xi, 2 * xi - xk, xk, fi, fj, fk, low[i], up[i])
        end
    end
    return (dim == 1) ? L[1] : L
end

"""
# References:

- Zhao, Weiguo, Liying Wang, Zhenxing Zhang, Seyedali Mirjalili, Nima Khodadadi, and Qiang Ge.
  "Quadratic Interpolation Optimization (QIO): A new optimization algorithm based on generalized quadratic interpolation and its applications to real-world engineering problems."
  Computer Methods in Applied Mechanics and Engineering 417 (2023): 116446.
"""
function QIO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return QIO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function QIO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    IndividualsPos = zeros(npop, dim)
    IndividualsFit = zeros(npop)

    if length(lb) == 1
        lb = ones(dim) .* lb
        ub = ones(dim) .* ub
    end

    IndividualsPos = initialization(npop, dim, lb, ub)

    for i = 1:npop
        # IndividualsPos[i, :] .= rand(dim) .* (ub .- lb) .+ lb
        IndividualsFit[i] = objfun(IndividualsPos[i, :])
    end

    BestF = Inf
    BestX = []

    for i = 1:npop
        if IndividualsFit[i] <= BestF
            BestF = IndividualsFit[i]
            BestX = IndividualsPos[i, :]
        end
    end

    HisBestF = zeros(max_iter)

    for It = 1:max_iter
        newIndividualPos = zeros(dim)

        for i = 1:npop
            if rand() > 0.5
                K = vcat(1:i-1, i+1:npop)
                RandInd = randperm(npop - 1)[1:3]
                K1 = K[RandInd[1]]
                K2 = K[RandInd[2]]
                K3 = K[RandInd[3]]
                f1 = IndividualsFit[i]
                f2 = IndividualsFit[K1]
                f3 = IndividualsFit[K2]

                for j = 1:dim
                    x1 = IndividualsPos[i, j]
                    x2 = IndividualsPos[K1, j]
                    x3 = IndividualsPos[K2, j]
                    newIndividualPos[j] = GQI(x1, x2, x3, f1, f2, f3, lb[j], ub[j])
                end

                a = cos(pi / 2 * It / max_iter)
                b = 0.7 * a + 0.15 * a * (cos(5 * pi * It / max_iter) + 1)
                w1 = 3 * b * randn()
                newIndividualPos .+= w1 .* (IndividualsPos[K3, :] .- newIndividualPos) .+ round(0.5 * (0.05 + rand())) * (log(rand() / rand()))

            else
                K = vcat(1:i-1, i+1:npop)
                RandInd = randperm(npop - 1)[1:2]
                K1 = K[RandInd[1]]
                K2 = K[RandInd[2]]
                f1 = IndividualsFit[K1]
                f2 = IndividualsFit[K2]
                f3 = BestF

                for j = 1:dim
                    x1 = IndividualsPos[K[RandInd[1]], j]
                    x2 = IndividualsPos[K[RandInd[2]], j]
                    x3 = BestX[j]
                    newIndividualPos[j] = GQI(x1, x2, x3, f1, f2, f3, lb[j], ub[j])
                end

                w2 = 3 * (1 - (It - 1) / max_iter) * randn()
                rD = rand(1:dim)
                newIndividualPos .+= w2 * (BestX .- round(1 + rand()) * (ub .- lb) ./ (ub[rD] - lb[rD]) .* IndividualsPos[i, rD])
            end

            newIndividualPos = clamp.(newIndividualPos, lb, ub)
            newIndividualFit = objfun(newIndividualPos)
            if newIndividualFit < IndividualsFit[i]
                IndividualsFit[i] = newIndividualFit
                IndividualsPos[i, :] .= newIndividualPos
            end
        end

        for i = 1:npop
            if IndividualsFit[i] < BestF
                BestF = IndividualsFit[i]
                BestX .= IndividualsPos[i, :]
            end
        end

        HisBestF[It] = BestF
    end

    # return BestF, BestX, HisBestF
    return OptimizationResult(
        BestX,
        BestF,
        HisBestF)
end

function QIO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return QIO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end