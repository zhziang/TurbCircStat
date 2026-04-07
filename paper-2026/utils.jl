using FourierFlows, CUDA

"""
	Compute the velocity circulation over a rectangular loop fixed by 'width' and 'height'
	using the convolution of the vorcity field 'ζ' and the loop Heaviside 'hs'.
	'ζh' and 'hsh' are their Fourier transforms respectively.
"""
function rectCirculations(ζh, width::Int, height::Int, grid)
	hsh = begin
        @devzeros typeof(grid.device) eltype(grid) (grid.nx, grid.ny) hs
		hs[1:width,1:height] .= 1
		grid.rfftplan * hs
	end

	Γh = device_array(grid)(ζh .* hsh) * (grid.dx * grid.dy)
	return grid.rfftplan \ Γh
end

" Compute the radial spectrum of enstrophy."
function enstrophySpectrum(ζh, grid)
	Ωh = device_array(grid)(abs2.(ζh) / (grid.nx * grid.ny)^2 / 2)
	return FourierFlows.radialspectrum(Ωh, grid; refinement = 1)
end

" Compute the radial spectrum of energy."
function energySpectrum(ζh, grid)
	Ωh = device_array(grid)(abs2.(ζh) .* grid.invKrsq / (grid.nx * grid.ny)^2 / 2)
	return FourierFlows.radialspectrum(Ωh, grid; refinement = 1)
end

" Compute the radial flux of enstrophy."
function radialEnstrophyFlux(ζh, grid)
	ζx = grid.rfftplan \ (@. im * grid.kr * ζh)
	ζy = grid.rfftplan \ (@. im * grid.l * ζh)
	u = grid.rfftplan \ (@. im * grid.l * grid.invKrsq * ζh)
	v = grid.rfftplan \ (@. -im * grid.kr * grid.invKrsq * ζh)
	Nh = grid.rfftplan * (@. u * ζx + v * ζy)
	fh = real.(Nh .* conj(ζh)) / (grid.nx * grid.ny)^2
	kr, fhr = FourierFlows.radialspectrum(fh, grid; refinement = 1)
	return (kr, cumsum(vec(fhr)) .* step(kr))
end

" Compute the radial flux of energy."
function radialEnergyFlux(ζh, grid)
	ζx = grid.rfftplan \ (@. im * grid.kr * ζh)
	ζy = grid.rfftplan \ (@. im * grid.l * ζh)
	u = grid.rfftplan \ (@. im * grid.l * grid.invKrsq * ζh)
	v = grid.rfftplan \ (@. -im * grid.kr * grid.invKrsq * ζh)
	Nh = grid.rfftplan * (@. u * ζx + v * ζy)
	fh = real.(Nh .* conj(ζh) .* grid.invKrsq) / (grid.nx * grid.ny)^2
	kr, fhr = FourierFlows.radialspectrum(fh, grid; refinement = 1)
	return (kr, cumsum(vec(fhr)) .* step(kr))
end

" Estimate the probability distribution function from a given array of samples (CPU ver.) "
function getPDF(samples::Array, bin)
	hist = zeros(length(bin)-1)

	for sp in samples
		idx = floor(Int, (sp - first(bin)) / step(bin)) + 1
		if 1 ≤ idx ≤ length(bin) - 1
			hist[idx] += 1
		end
	end

	return hist / (step(bin) * length(samples))
end

" Estimate the probability distribution function from a given array of samples (GPU ver.) "
function getPDF(samples::CuArray, bin)
	hist = CUDA.zeros(length(bin)-1)

	threads_per_block = 256

	blocks_per_grid = cld(length(samples), threads_per_block)

	@cuda threads = threads_per_block blocks = blocks_per_grid histogram_kernel!(
		hist, samples, first(bin), step(bin), length(bin)-1,
	)

	return hist / (step(bin) * length(samples))
end

function histogram_kernel!(hist, data, min_val, bin_width, nbins)
	idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	if idx <= length(data)
		value = data[idx]
		if value >= min_val && value <= min_val + bin_width * nbins
			bin_idx = Int(floor((value - min_val) / bin_width)) + 1
			bin_idx = max(1, min(bin_idx, nbins))
			CUDA.@atomic hist[bin_idx] += 1
		end
	end
	return nothing
end
