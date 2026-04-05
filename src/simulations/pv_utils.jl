function hamiltonian(x, y, Γ)
	n = length(x)
	@. x = mod(x, 1)
	@. y = mod(y, 1)
	z = x .+ 1im * y
	diffz = z .- transpose(z)

	H = Γ .* Hᵢⱼ.(diffz) .* Γ'

	return sum(H[(1:n) .< (1:n)'])
end

function velocity(x, y, Γ)
	n = length(x)
	@. x = mod(x, 1)
	@. y = mod(y, 1)
	z = x .+ 1im * y
	diffz = z .- transpose(z)

	U = Uᵢⱼ.(diffz) .* Γ'
	U[(1:n) .== (1:n)'] .= 0

	cxVel = sum(U; dims = 2)
	return (real(cxVel), imag(cxVel))
end


@inline Uᵢⱼ(z) = 1im * (conj(wζ(z)) - π * z) / (2π)

@inline Hᵢⱼ(z) = - (real(log(wσ(z))) - π * abs2(z) / 2) / (2π)

const dϑ₁0 = 2 * exp(-π/4) * sum(k -> (-1)^k * exp(-π)^(k*(k+1)) * (2k+1), 0:10)
const d³ϑ₁0 = 2 * exp(-π/4) * sum(k -> -(-1)^k * exp(-π)^(k*(k+1)) * (2k+1)^3, 0:10)
const η₁ = - π^2 * d³ϑ₁0/(6 * dϑ₁0)

@inline wζ(z) = 2 * η₁ * z + π * dϑ₁(π*z) / (ϑ₁(π*z))

@inline wσ(z) = exp(η₁ * z ^ 2) * ϑ₁(π*z) / (π * dϑ₁0)

@inline function ϑ₁(z)
	res = zero(z)
	for k in 0:10
		res += (-1)^k * exp(-π)^(k*(k+1)) * sin((2k+1)*z)
	end
	return 2 * exp(-π/4) * res
end

@inline function dϑ₁(z)
	res = zero(z)
	for k in 0:10
		res += (-1)^k * exp(-π)^(k*(k+1)) * (2k+1) * cos((2k+1)*z)
	end
	return 2 * exp(-π/4) * res
end

@inline function d³ϑ₁(z)
	res = zero(z)
	for k in 0:10
		res += -(-1)^k * exp(-π)^(k*(k+1)) * (2k+1)^3 * cos((2k+1)*z)
	end
	return 2 * exp(-π/4) * res
end


