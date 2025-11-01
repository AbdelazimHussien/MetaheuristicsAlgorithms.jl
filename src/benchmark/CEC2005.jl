using LinearAlgebra

# Ufun used in F12 and F13
function Ufun(x, a, k, m)
    return k .* ((x .- a).^m) .* (x .> a) + k .* ((-x .- a).^m) .* (x .< -a)
end

"""
    sphere(x)

Sphere function.

A basic unimodal benchmark used in optimization.

# Equation
```math
f(\\mathbf{x}) = \\sum_{i=1}^n x_i^2
```

# Properties
- Domain: any dimension n; typical bounds xᵢ ∈ [-100, 100].
- Global minimum: `f(0,…,0) = 0` at `x = 0`.
- Complex support: if x contains complex or dual numbers, this implementation uses abs2(xᵢ) (the squared modulus), so the result is a real, non-negative scalar.
"""
sphere(x) = sum(abs2, x)

const F1 = sphere

"""
    schwefel_2_22(x)

Schwefel’s Problem 2.22 (F2).

A basic benchmark that combines the sum and the product of absolute values. Often used
to test robustness to scaling and zero/near-zero components.

# Equation
```math
f(\\mathbf{x}) = \\sum_{i=1}^n |x_i| \\;+\\; \\prod_{i=1}^n |x_i|
````

# Properties
- Domain: any dimension `n`; typical bounds `xᵢ ∈ [-10, 10]` (variants exist).
- Global minimum: `f(0,…,0) = 0` at `x = 0`.
- Complex/AD support: this implementation uses `abs(xᵢ)` (modulus), so it works with
  complex and dual numbers; the result is a real, non-negative scalar.
- Note: if any `xᵢ = 0`, the product term is `0`.
"""
schwefel_2_22(x) = sum(abs, x) + prod(abs, x)

const F2 = schwefel_2_22

"""
    schwefel_1_2(x)

Schwefel’s Problem 1.2 (F3).

Cumulative-sum-of-squares benchmark.

This test function computes the sum of the squares of cumulative sums of the input vector.
It increases the dependency between variables and is used to test an algorithm’s ability
to handle variable interactions.

# Equation
```math
f(\\mathbf{x}) = \\sum_{i=1}^n \\left( \\sum_{j=1}^i x_j \\right)^2
```

# Properties
* Domain: any dimension `n`; typical bounds `xᵢ ∈ [-100, 100]`.
* Global minimum: `f(0,…,0) = 0` at `x = 0`.
* Structure: convex, **non-separable**, unimodal; conditioning worsens with `n`.
* Complexity: implemented in O(n) via a single pass of prefix sums (no nested sums).
* Complex/AD support: uses `abs2(prefix)` so the result is a real, non-negative scalar
  for complex and dual-number inputs; for real `x` this equals the textbook definition.
"""
function schwefel_1_2(x)
    T = float(eltype(x))
    s, o = zero(T), zero(T)
    @inbounds @simd for xi in x
        s += xi
        o += abs2(s)
    end
    return o
end

const F3 = schwefel_1_2

"""
    schwefel_2_21(x)

Schwefel’s Problem 2.21 (F4).

Maximum Absolute Value Function.

This function returns the maximum of the absolute values of the input vector elements.
It is used to test an optimizer's ability to minimize the worst-case (largest-magnitude) variable.

# Equation
```math
f(\\mathbf{x}) = \\max_{1 \\le i \\le n} |x_i|
```

# Properties
* Domain: any dimension `n`; typical bounds `xᵢ ∈ [-100, 100]`.
* Global minimum: `f(0,…,0) = 0` at `x = 0`.
* Complex/AD support: uses `abs(xᵢ)` (modulus), so it works with complex and dual numbers; the result is a real, non-negative scalar.
"""
schwefel_2_21(x) = maximum(abs, x)

const F4 = schwefel_2_21

"""
    rosenbrock(x)

Rosenbrock Function (F5).

A classic, non-convex test problem for optimization algorithms. It has a narrow, curved
valley leading to the global minimum, which makes convergence difficult.

# Equation
```math
f(\\mathbf{x}) = \\sum_{i=1}^{n-1} \\left[ 100\\,(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \\right]
```

# Properties
* Domain: any dimension `n`; typical bounds `xᵢ ∈ [-30, 30]`.
* Global minimum: `f(1,…,1) = 0` at `x = (1,…,1)`.
* Structure: non-convex, non-separable; narrow, curved valley.
* Numerical: implemented in **O(n)** with a single pass and no intermediate allocations.
* AD support: works with dual numbers; intended for real-valued inputs.
"""
function rosenbrock(x)
    T = float(eltype(x))
    n = length(x)
    @assert n >= 2 "Rosenbrock requires length(x) ≥ 2"
    s = zero(T)
    @inbounds @simd for i = 1:(n-1)
        xi   = x[i]
        xip1 = x[i+1]
        s    += 100 * (xip1 - xi*xi)^2 + (xi - 1)^2
    end
    return s
end

const F5 = rosenbrock

"""
    F6(x)

Shifted Sphere Function.

This function is a variation of the Sphere function where each variable is shifted by 0.5 before squaring. It remains unimodal but shifts the global minimum from the origin.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n (x_i + 0.5)^2
```
"""
F6(x) = sum(abs.(x .+ 0.5).^2)

"""
    F7(x)

Weighted Quartic Function with Noise.

This function adds a random noise term to a weighted sum of the fourth powers of the input variables. The noise introduces stochasticity, making it useful for testing robustness of optimization algorithms.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n i \\cdot x_i^4 + \\text{rand}()
```
"""
function F7(x)
    dim = length(x)
    return sum((1:dim) .* (x.^4)) + rand()
end

"""
    F8(x)

Schwefel Function.

A widely used multimodal benchmark function with many local minima. It poses a challenge for optimization
    algorithms due to its deceptive landscape and large search space.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n -x_i \\cdot \\sin(\\sqrt{|x_i|})
```
"""
F8(x) = sum(-x .* sin.(sqrt.(abs.(x))))

"""
    F9(x)

Rastrigin Function.

A highly multimodal benchmark function commonly used to evaluate the performance of global optimization algorithms. Its large number of local minima makes it particularly challenging.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n \\left( x_i^2 - 10 \\cos(2\\pi x_i) + 10 \\right)
```
"""
function F9(x)
    dim = length(x)
    return sum(x.^2 .- 10 .* cos.(2π .* x)) + 10 * dim
end

"""
    F10(x)

Ackley Function.

A popular multimodal benchmark function used to test optimization algorithms.
It features a nearly flat outer region and a large number of local minima, making convergence difficult.

# Equation

```math
f(\\mathbf{x}) = -20 \\exp\\left(-0.2 \\sqrt{\\frac{1}{n} \\sum_{i=1}^n x_i^2}\\right)
                - \\exp\\left(\\frac{1}{n} \\sum_{i=1}^n \\cos(2\\pi x_i)\\right)
                + 20 + e
```
"""
function F10(x)
    dim = length(x)
    return -20 * exp(-0.2 * sqrt(sum(x.^2) / dim)) -
           exp(sum(cos.(2π .* x)) / dim) + 20 + exp(1)
end

"""
    F11(x)

Griewank Function.

A widely used multimodal test function with many widespread local minima, but a simple global minimum at the origin.

# Equation

```math
f(\\mathbf{x}) = \\frac{1}{4000} \\sum_{i=1}^n x_i^2 - \\prod_{i=1}^n \\cos\\left( \\frac{x_i}{\\sqrt{i}} \\right) + 1
```
"""
function F11(x)
    dim = length(x)
    return sum(x.^2) / 4000 - prod(cos.(x ./ sqrt.(1:dim))) + 1
end

"""
    F12(x)

Penalized Function #1.

A multimodal benchmark function with penalization terms to enforce constraints, often used in optimization testing.

# Equation

```math
f(\\mathbf{x}) = \\frac{\\pi}{n} \\left[ 10 \\sin^2 \\left( \\pi \\left(1 + \\frac{x_1 + 1}{4} \\right) \\right) + \\sum_{i=1}^{n-1} \\left( \\frac{x_i + 1}{4} \\right)^2 \\left( 1 + 10 \\sin^2 \\left( \\pi \\left(1 + \\frac{x_{i+1} + 1}{4} \\right) \\right) \\right) + \\left( \\frac{x_n + 1}{4} \\right)^2 \\right] + \\sum_{i=1}^n U(x_i, 10, 100, 4)
```
"""
function F12(x)
    dim = length(x)
    term1 = (π / dim) * (10 * sin(π * (1 + (x[1] + 1) / 4))^2)
    term2 = sum(((x[1:dim-1] .+ 1) ./ 4).^2 .* (1 .+ 10 .* sin.(π .* (1 .+ (x[2:dim] .+ 1) ./ 4)).^2))
    term3 = ((x[dim] + 1) / 4)^2
    return term1 + term2 + term3 + sum(Ufun(x, 10, 100, 4))
end


"""
    F13(x)

Penalized Function #2.

A multimodal benchmark function with penalization terms used to test optimization algorithms, featuring sine and quadratic terms.

# Equation

```math
f(\\mathbf{x}) = 0.1 \\left[ \\sin^2(3 \\pi x_1) + \\sum_{i=1}^{n-1} (x_i - 1)^2 (1 + \\sin^2(3 \\pi x_{i+1})) + (x_n - 1)^2 (1 + \\sin^2(2 \\pi x_n)) \\right] + \\sum_{i=1}^n U(x_i, 5, 100, 4)
```
"""
function F13(x)
    dim = length(x)
    term1 = sin(3π * x[1])^2
    term2 = sum((x[1:dim-1] .- 1).^2 .* (1 .+ sin.(3π .* x[2:dim]).^2))
    term3 = (x[dim] - 1)^2 * (1 + sin(2π * x[dim])^2)
    return 0.1 * (term1 + term2 + term3) + sum(Ufun(x, 5, 100, 4))
end


"""
    F14(x)

Shekel's Foxholes Function.

A challenging multimodal benchmark function used to test optimization algorithms. The function has many local minima, making it useful for assessing global search capability.

# Equation

```math
f(\\mathbf{x}) = \\left[ 0.002 + \\sum_{j=1}^{25} \\frac{1}{j + \\sum_{i=1}^{2} (x_i - a_{ij})^6} \\right]^{-1}
```
"""
function F14(x)
    # println("length(x) ", length(x))
    # println("(x) ", x)
    @assert length(x) == 2 "F14 expects x to be a 2D vector"
    aS = hcat([-32, -16, 0, 16, 32,
               -32, -16, 0, 16, 32,
               -32, -16, 0, 16, 32,
               -32, -16, 0, 16, 32,
               -32, -16, 0, 16, 32]...,
              [-32, -32, -32, -32, -32,
               -16, -16, -16, -16, -16,
                0,  0,  0,  0,  0,
               16, 16, 16, 16, 16,
               32, 32, 32, 32, 32]...)
    bS = [sum((x .- aS[:, j]).^6) for j in 1:25]
    # return (1 / (0.002 + sum(1 ./ (1:25 .+ bS))))^(-1)
    return (1 / (0.002 + sum(1 ./ ((1:25) .+ bS))))^(-1)
end

"""
    F15(x)

Kowalik and Osborne Function.

A nonlinear least squares problem used in parameter estimation and optimization. It is known for its narrow, curved valley structure, which poses a challenge for optimization algorithms.

# Equation

```math
f(\\mathbf{x}) = \\sum_{k=1}^{11} \\left[ a_k - \\frac{x_1 (b_k^2 + x_2 b_k)}{b_k^2 + x_3 b_k + x_4} \\right]^2
```
"""
function F15(x)
    aK = [.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246]
    bK = 1 ./ [.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    return sum((aK .- (x[1] .* (bK.^2 .+ x[2] .* bK)) ./ (bK.^2 .+ x[3] .* bK .+ x[4])).^2)
end

"""
    F16(x)

Six-Hump Camel Function.

A well-known multimodal benchmark function with six local minima, two of which are global. Often used to evaluate global optimization algorithms.

# Equation

```math
f(x_1, x_2) = 4x_1^2 - 2.1x_1^4 + \\frac{1}{3}x_1^6 + x_1 x_2 - 4x_2^2 + 4x_2^4
```
"""
function F16(x)
    return 4*x[1]^2 - 2.1*x[1]^4 + (x[1]^6)/3 + x[1]*x[2] - 4*x[2]^2 + 4*x[2]^4
end

"""
    F17(x)

Branin Function (also known as Branin-Hoo Function).

A widely used benchmark function for optimization algorithms. It has multiple global minima and is commonly used for testing the performance of global optimizers in 2D.

# Equation

```math
f(x_1, x_2) = \\left(x_2 - \\frac{5.1}{4\\pi^2}x_1^2 + \\frac{5}{\\pi}x_1 - 6\\right)^2 + 10 \\left(1 - \\frac{1}{8\\pi}\\right)\\cos(x_1) + 10
```
"""
function F17(x)
    return (x[2] - x[1]^2 * 5.1 / (4π^2) + 5 / π * x[1] - 6)^2 +
           10 * (1 - 1 / (8π)) * cos(x[1]) + 10
end

"""
    F18(x)

Goldstein–Price Function.

A classic two-dimensional test function for global optimization with a complex landscape containing many local minima and a known global minimum.

# Equation

```math
f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 (19 - 14x_1 + 3x_1^2 - 14x_2 + 6x_1x_2 + 3x_2^2)] \\\\
\\quad\\times [30 + (2x_1 - 3x_2)^2 (18 - 32x_1 + 12x_1^2 + 48x_2 - 36x_1x_2 + 27x_2^2)]
```
"""
function F18(x)
    return (1 + (x[1] + x[2] + 1)^2 * (19 - 14x[1] + 3x[1]^2 - 14x[2] +
            6x[1]*x[2] + 3x[2]^2)) * (30 + (2x[1] - 3x[2])^2 * (18 - 32x[1] +
            12x[1]^2 + 48x[2] - 36x[1]*x[2] + 27x[2]^2))
end

"""
    F19(x)

Hartmann 3D Function.

A common multimodal benchmark function used to test the performance of global optimization algorithms in 3 dimensions. It is characterized by several local minima and one known global minimum.

# Equation

```math
f(\\mathbf{x}) = -\\sum_{i=1}^4 c_i \\exp\\left(-\\sum_{j=1}^3 a_{ij} (x_j - p_{ij})^2\\right)
```
"""
function F19(x)
    aH = [3 10 30; .1 10 35; 3 10 30; .1 10 35]
    cH = [1, 1.2, 3, 3.2]
    pH = [.3689 .1170 .2673;
          .4699 .4387 .7470;
          .1091 .8732 .5547;
          .0381 .5743 .8828]
    return -sum(cH[i] * exp(-sum(aH[i, :] .* (x .- pH[i, :]).^2)) for i in 1:4)
end

"""
    F20(x)

Hartmann 6D Function.

A widely used multimodal benchmark function in 6 dimensions for testing the performance of global optimization algorithms. It features a complex landscape with several local minima and a known global minimum.

# Equation

```math
f(\\mathbf{x}) = -\\sum_{i=1}^4 c_i \\exp\\left(-\\sum_{j=1}^6 a_{ij} (x_j - p_{ij})^2\\right)
```
"""
function F20(x)
    aH = [10 3 17 3.5 1.7 8;
          .05 10 17 .1 8 14;
          3 3.5 1.7 10 17 8;
          17 8 .05 10 .1 14]
    cH = [1, 1.2, 3, 3.2]
    pH = [.1312 .1696 .5569 .0124 .8283 .5886;
          .2329 .4135 .8307 .3736 .1004 .9991;
          .2348 .1415 .3522 .2883 .3047 .6650;
          .4047 .8828 .8732 .5743 .1091 .0381]
    return -sum(cH[i] * exp(-sum(aH[i, :] .* (x .- pH[i, :]).^2)) for i in 1:4)
end

"""
    F21(x)

Shekel’s Foxholes Function (m = 5).

A multimodal benchmark function often used to test optimization algorithms' ability to avoid local optima. This version uses `m = 5` terms in the summation.

# Equation

```math
f(\\mathbf{x}) = -\\sum_{i=1}^m \\left[ \\sum_{j=1}^4 (x_j - a_{ij})^2 + c_i \\right]^{-1}
```
"""
function F21(x)
    aSH = [4 4 4 4;
           1 1 1 1;
           8 8 8 8;
           6 6 6 6;
           3 7 3 7;
           2 9 2 9;
           5 5 3 3;
           8 1 8 1;
           6 2 6 2;
           7 3.6 7 3.6]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    return -sum((dot(x .- aSH[i, :], x .- aSH[i, :]) + cSH[i])^(-1) for i in 1:5)
end

"""
    F22(x)

Shekel’s Foxholes Function (m = 7).

A multimodal benchmark function commonly used for testing the ability of optimization algorithms to navigate complex landscapes with many local minima. This is a variant of the Shekel function with `m = 7` terms.

# Equation

```math
f(\\mathbf{x}) = -\\sum_{i=1}^m \\left[ \\sum_{j=1}^4 (x_j - a_{ij})^2 + c_i \\right]^{-1}
```
"""
function F22(x)
    aSH = [4 4 4 4;
           1 1 1 1;
           8 8 8 8;
           6 6 6 6;
           3 7 3 7;
           2 9 2 9;
           5 5 3 3;
           8 1 8 1;
           6 2 6 2;
           7 3.6 7 3.6]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    return -sum((dot(x .- aSH[i, :], x .- aSH[i, :]) + cSH[i])^(-1) for i in 1:7)
end

"""
    F23(x)

Shekel’s Foxholes Function (m = 10).

A classic multimodal benchmark function designed to test an optimization algorithm’s ability to avoid numerous local optima and find the global minimum. This version uses `m = 10` terms in the summation.

# Equation

```math
f(\\mathbf{x}) = -\\sum_{i=1}^{10} \\left[ \\sum_{j=1}^4 (x_j - a_{ij})^2 + c_i \\right]^{-1}
```

"""
function F23(x)
    aSH = [4 4 4 4;
           1 1 1 1;
           8 8 8 8;
           6 6 6 6;
           3 7 3 7;
           2 9 2 9;
           5 5 3 3;
           8 1 8 1;
           6 2 6 2;
           7 3.6 7 3.6]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    return -sum((dot(x .- aSH[i, :], x .- aSH[i, :]) + cSH[i])^(-1) for i in 1:10)
end
