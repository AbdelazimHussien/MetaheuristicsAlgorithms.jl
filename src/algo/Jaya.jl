"""
Rao, R. 
"Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems." 
International Journal of Industrial Engineering Computations 7, no. 1 (2016): 19-34.
"""


function Jaya(pop, maxGen, mini, maxi, var, objective)

    # Parameters
    # pop = 1000                # Population size
    # var = 10                  # Number of design variables
    # maxGen = 3000             # Maximum number of iterations
    # mini = -100 * ones(var)   # Lower Bound of Variables
    # maxi = 100 * ones(var)    # Upper Bound of Variables
    # objective = Sphere        # Cost Function (assumes a function named `Sphere` is defined)

    # Initialize
    x = zeros(pop, var)
    fnew = zeros(pop)
    f = zeros(pop)
    convergence = zeros(maxGen)
    fopt = zeros(maxGen)
    xopt = zeros(maxGen, var)

    if length(mini) == 1
        mini = mini * ones(var)
    end
    
    if length(maxi) == 1
        maxi = maxi * ones(var)
    end

    # Generate and Initialize the positions
    for i in 1:var
        x[:, i] = mini[i] .+ (maxi[i] - mini[i]) .* rand(pop)
    end

    for i in 1:pop
        f[i] = objective(x[i, :])
    end

    # Main Loop
    gen = 1
    while gen < maxGen
        row, col = size(x)
        t, tindex = findmin(f)
        Best = x[tindex, :]
        w, windex = findmax(f)
        worst = x[windex, :]
        xnew = zeros(row, col)
        
        for i in 1:row
            for j in 1:col
                xnew[i, j] = x[i, j] + rand() * (Best[j] - abs(x[i, j])) - (worst[j] - abs(x[i, j]))
            end
        end 
        
        for i in 1:row
            xnew[i, :] = clamp.(xnew[i, :], mini, maxi)
            fnew[i] = objective(xnew[i, :])
        end
        
        for i in 1:pop
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
    Fes = pop * ind
    # @printf("Optimum value = %.10f\n", val)

    # Plotting
    # using Plots
    # plot(fopt, linewidth=2, label="JAYA", xlabel="Iteration", ylabel="Best Cost", legend=:topright)

    best_value = minimum(fopt)
    best_position = xopt[argmin(fopt), :]

    return  best_value, best_position, convergence

end