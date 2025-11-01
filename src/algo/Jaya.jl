"""
# References:

- Rao, R.
  "Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems."
  International Journal of Industrial Engineering Computations 7, no. 1 (2016): 19-34.
"""
function Jaya(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return Jaya(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function Jaya(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)::OptimizationResult
    dim = length(lb)

    # Parameters
    # npop = 1000                # Population size
    # var = 10                  # Number of design variables
    # max_iter = 3000             # Maximum number of iterations
    # mini = -100 * ones(var)   # Lower Bound of Variables
    # maxi = 100 * ones(var)    # Upper Bound of Variables
    # objfun = Sphere        # Cost Function (assumes a function named `Sphere` is defined)

    # Initialize
    x = zeros(npop, dim)
    fnew = zeros(npop)
    f = zeros(npop)
    convergence = zeros(max_iter)
    fopt = zeros(max_iter)
    xopt = zeros(max_iter, dim)

    if length(lb) == 1
        lb = lb * ones(dim)
    end

    if length(ub) == 1
        ub = ub * ones(dim)
    end

    # Generate and Initialize the positions
    x = initializationLogistic(npop, dim, ub, lb)
    # for i = 1:dim
    #     x[:, i] = lb[i] .+ (ub[i] - lb[i]) .* rand(npop)
    # end

    for i = 1:npop
        f[i] = objfun(x[i, :])
    end

    # Main Loop
    gen = 1
    while gen < max_iter
        row, col = size(x)
        t, tindex = findmin(f)
        Best = x[tindex, :]
        w, windex = findmax(f)
        worst = x[windex, :]
        xnew = zeros(row, col)

        for i = 1:row
            for j = 1:col
                xnew[i, j] = x[i, j] + rand() * (Best[j] - abs(x[i, j])) - (worst[j] - abs(x[i, j]))
            end
        end

        for i = 1:row
            xnew[i, :] = clamp.(xnew[i, :], lb, ub)
            fnew[i] = objfun(xnew[i, :])
        end

        for i = 1:npop
            if fnew[i] < f[i]
                x[i, :] = xnew[i, :]
                f[i] = fnew[i]
            end
        end

        fopt[gen] = minimum(f)
        xopt[gen, :] = x[argmin(f), :]
        gen += 1
        # @printf("Iteration No. = %d, Best Cost = %.4f\n", gen, fopt[gen-1])
        convergence[gen] = minimum(f)
    end

    # Results
    val, ind = findmin(fopt)
    Fes = npop * ind
    # @printf("Optimum value = %.10f\n", val)

    # Plotting
    # using Plots
    # plot(fopt, linewidth=2, label="JAYA", xlabel="Iteration", ylabel="Best Cost", legend=:topright)

    best_value = minimum(fopt)
    best_position = xopt[argmin(fopt), :]

    # return best_value, best_position, convergence
    return OptimizationResult(
        best_position,
        best_value,
        convergence)
end

function Jaya(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return Jaya(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end