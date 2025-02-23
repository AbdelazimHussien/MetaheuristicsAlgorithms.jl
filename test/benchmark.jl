function benchmark(xx; funNo=1)
    npop = size(xx, 2)
    y = Vector{eltype(xx)}(undef, npop)

    for i = 1:npop
        x1 = xx[1, i]
        if size(xx, 1) >= 2
            x2 = xx[2, i]
        end
        if funNo == 1 # EGGHOLDER FUNCTION
            term1 = -(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47)))
            term2 = -x1 * sin(sqrt(abs(x1 - (x2 + 47))))
            y[i] = term1 + term2
            # f(x⁺) = -959.6407, at x⁺ = [512, 404.2319]

        elseif funNo == 2 # CROSS-IN-TRAY FUNCTION
            fact1 = sin(x1) * sin(x2)
            fact2 = exp(abs(100 - sqrt(x1^2 + x2^2) / pi))
            y[i] = -0.0001 * (abs(fact1 * fact2) + 1)^0.1
        elseif funNo == 3 # DROP-WAVE FUNCTION
            frac1 = 1 + cos(12 * sqrt(x1^2 + x2^2))
            frac2 = 0.5 * (x1^2 + x2^2) + 2
            y[i] = -frac1 / frac2
        elseif funNo == 4 # GRIEWANK FUNCTION
            D = size(xx, 2)
            prod = cumprod(cos(xx(i, :) ./ sqrt(1:D)))
            y[i] = sum(xx(:, i) .^ 2 / 4000) - prod[end] + 1
        end
    end

    return y
end