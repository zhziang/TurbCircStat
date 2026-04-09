
" The mutually induced velocity for a given configuration of point vortices. "
@inline hvec(x, y) = begin
	z = x + 1im*y
	c = 1im * (conj(wζ(z)) - π * z) / (2π)
	real(c), imag(c)
end

@inline green(x, y) = begin
	z = x + 1im*y
	- (real(log(wσ(z))) - π * abs2(z) / 2) / (2π)
end

# Realization of special functions

" Weierstrass's zeta function. "
@inline wζ(z) = 2 * η₁ * z + π * dϑ₁(π*z) / (ϑ₁(π*z))

" Weierstrass's sigma function. "
@inline wσ(z) = exp(η₁ * z ^ 2) * ϑ₁(π*z) / (π * dϑ₁0)

@inline ϑ₁(z) = begin
	result = 0.0
    for k in 0:10
        result += (-1)^k * exp(-π)^(k*(k+1)) * sin((2k+1)*z)
    end
    return 2 * exp(-π/4) * result
end

@inline dϑ₁(z) = begin
	result = 0.0
    for k in 0:10
        result += (-1)^k * exp(-π)^(k*(k+1)) * (2k+1) * cos((2k+1)*z)
    end
    return 2 * exp(-π/4) * result
end

@inline d³ϑ₁(z) = begin
	result = 0.0
    for k in 0:10
        result += -(-1)^k * exp(-π)^(k*(k+1)) * (2k+1)^3 * cos((2k+1)*z)
    end
    return 2 * exp(-π/4) * result
end

const dϑ₁0 = dϑ₁(0)
const d³ϑ₁0 = d³ϑ₁(0)
const η₁ = - π^2 * d³ϑ₁0/(6 * dϑ₁0)


