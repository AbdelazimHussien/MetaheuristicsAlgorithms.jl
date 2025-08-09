# Tutorial
In this Example, we try to solve **Ackley function**
```math
f(\mathbf{x}) = -a \exp\left(-b \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}\right)
               - \exp\left(\frac{1}{n} \sum_{i=1}^{n} \cos(c x_i)\right)
               + a + \exp(1)
```
with ``a = 20``, ``b = 0.2``, ``c = 2π``, and global minimum at ``x=0``.

```@example
using MetaheuristicsAlgorithms

function Ackley(x::Vector{Float64})::Float64
    n = length(x)
    a = 20.0
    b = 0.2
    c = 2π

    sum1 = sum(xi^2 for xi in x)
    sum2 = sum(cos(c * xi) for xi in x)

    term1 = -a * exp(-b * sqrt(sum1 / n))
    term2 = -exp(sum2 / n)

    return term1 + term2 + a + exp(1)
end
lb = [-32.768 for i = 1:5]
ub = [32.768 for i = 1:5]
result = AEO(Ackley, lb, ub, 100, 1000)
convergence_curve(result)
```

The package contains a lot of algorithms that the user can use to compare between them. The function called `compare_algorithms`. For example I will compare the following algorithms Artificial Ecosystem-based Optimization (AEO), Artificial lemming algorithm (ALA), Bald Eagle Search (BES), Black-winged kite algorithm (BKA), Butterfly Optimization Algorithm (BOA), Chimp Optimization Algorithm (ChOA)

```@example
using MetaheuristicsAlgorithms

lb = -100
ub = 100
problem = OptimizationProblem(Engineering_F1, lb, ub, 3)

algos = [AEO, ALA]
results = compare_algorithms(algos, problem; npop=30, max_iter=100, runs=30)
mean_fitnesses = Dict{String, Float64}()

using Plots
using Statistics
p1 = Plots.plot()
for (name, sols) in results
    curves = [sol.his_best_fit for sol in sols]
    max_len = maximum(length.(curves))
    padded_curves = [length(c) == max_len ? c : vcat(c, fill(c[end], max_len - length(c))) for c in curves]
    mean_curve = [mean(getindex.(padded_curves, i)) for i in 1:max_len]
    Plots.plot!(p1, mean_curve, label=name)
end

xlabel!("Iterations")
ylabel!("Objective Function Value")
title!("Mean Convergence Curves")
Plots.savefig(p1, "convergence.png")
# display(p1)
```
