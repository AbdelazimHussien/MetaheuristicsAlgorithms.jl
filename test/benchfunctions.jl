function Ackley(x::Vector{Float64})::Float64
    n = length(x)
    a = 20.0
    b = 0.2
    c = 2 * π

    sum1 = sum(xi^2 for xi in x)
    sum2 = sum(cos(c * xi) for xi in x)

    term1 = -a * exp(-b * sqrt(sum1 / n))
    term2 = -exp(sum2 / n)

    return term1 + term2 + a + exp(1)
end

function Griewank(x::Vector{Float64})::Float64
    sum_term = sum(xi^2 / 4000 for xi in x)
    prod_term = prod(cos(xi / sqrt(i + 1)) for (i, xi) in enumerate(x))
    return sum_term - prod_term + 1
end

"f(x⁺) = -959.6407, at x⁺ = [512, 404.2319]"
function eggholder(x::Vector{Float64})::Float64
    x1, x2 = x
    return -(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47))) - x1 * sin(sqrt(abs(x1 - (x2 + 47))))
end

"f(x⁺) = -2.06261, at x⁺ = [±1.34941, ±1.34941]"
function cross_in_tray(x::Vector{Float64})::Float64
    x1, x2 = x
    return -0.0001 * (abs(sin(x1) * sin(x2) * exp(abs(100 - sqrt(x1^2 + x2^2) / pi))) + 1)^0.1
end

"f(x⁺) = -1.03163, at x⁺ = [0, 0]"
function drop_wave(x::Vector{Float64})::Float64
    x1, x2 = x
    return -0.5 - cos(12 * sqrt(x1^2 + x2^2)) + 0.5 * (x1^2 + x2^2) + 2
end