using HDF5, Measurements
include("utils.jl")

srcPath = "./.output"
desPath = (@__DIR__)
dev = GPU()
ngrid = 1024
Reynolds = [3.5; 3.7; 3.9; 4.0; 4.02; 4.04; 4.05]

struct PostData
	" Source data (the vorticity fields). "
	ζhs::Any
	" The destination of the output data. "
	des::HDF5.Group
	" Method deriving the output data from the sourece data: method(ζhs,des)."
	method::Function
	function PostData(method, src::HDF5.File, des::HDF5.Group)
		ζhs = Iterators.map(src) do ds
			ζh = read(ds)
			device_array(dev)(ζh)
		end
		return new(ζhs, des, method)
	end
end

" Collect the output data. "
collect(pd::PostData) = pd.method(pd.ζhs, pd.des)


lpFilter(Re) = device_array(dev)(float(grid.Krsq .< (ngrid / 3sqrt(Re) - 5)^2))
grid = TwoDGrid(dev; nx = ngrid, Lx = 2π)
srcFiles = Dict([Re=>h5open(srcPath * "/Re$(Re)_N$(ngrid).h5", "r") for Re in Reynolds])
desFile = h5open(desPath * "/postdata.h5", "w")

energy_spectrum(Re) = PostData(
	(ζhs, des) -> begin
		kr, _ = energySpectrum(first(ζhs), grid)

		mean = sum(ζhs) do ζh
			_, Ehr = energySpectrum(ζh, grid)
			Ehr
		end ./ length(ζhs)

		var = sum(ζhs) do ζh
			_, Ehr = energySpectrum(ζh, grid)
			@. (Ehr - mean) ^ 2
		end ./ length(ζhs)

		des["x"] = Array(kr)
		des["y"] = Array(mean)
		des["Δy"] = Array(sqrt.(var))
	end,
	srcFiles[Re],
	create_group(desFile, "wavenumber-energy_spectra/$(Re)"),
)

energy_flux(Re) = PostData(
	(ζhs, des) -> begin
		kr, _ = radialEnergyFlux(first(ζhs), grid)

		mean = sum(ζhs) do ζh
			_, Ehr = radialEnergyFlux(ζh, grid)
			Ehr
		end ./ length(ζhs)

		var = sum(ζhs) do ζh
			_, Ehr = radialEnergyFlux(ζh, grid)
			@. (Ehr - mean) ^ 2
		end ./ length(ζhs)

		des["x"] = Array(kr)
		des["y"] = Array(mean)
		des["Δy"] = Array(sqrt.(var))
	end,
	srcFiles[Re],
	create_group(desFile, "wavenumber-energy_fluxes/$(Re)"),
)

enstrophy_flux(Re) = PostData(
	(ζhs, des) -> begin
		kr, _ = radialEnstrophyFlux(first(ζhs), grid)

		mean = sum(ζhs) do ζh
			_, Ehr = radialEnstrophyFlux(ζh, grid)
			Ehr
		end ./ length(ζhs)

		var = sum(ζhs) do ζh
			_, Ehr = radialEnstrophyFlux(ζh, grid)
			@. (Ehr - mean) ^ 2
		end ./ length(ζhs)

		des["x"] = Array(kr)
		des["y"] = Array(mean)
		des["Δy"] = Array(sqrt.(var))
	end,
	srcFiles[Re],
	create_group(desFile, "wavenumber-enstrophy_fluxes/$(Re)"),
)

large_scale_velocity_pdf(Re) = PostData(
	(ζhs, des) -> begin
		fh = lpFilter(Re)

		umax = maximum(ζhs) do ζh
			uh = im * ζh .* grid.invKrsq .* grid.l
			u = grid.rfftplan \ uh
			maximum(abs.(u))
		end

		bin = range(-umax, umax, 100)

		mean = sum(ζhs) do ζh
			uh = im * ζh .* grid.invKrsq .* grid.l
			u = grid.rfftplan \ uh
			getPDF(u, bin)
		end ./ length(ζhs)

		var = sum(ζhs) do ζh
			uh = im * ζh .* grid.invKrsq .* grid.l
			u = grid.rfftplan \ uh
			pdf = getPDF(u, bin)
            @. (pdf - mean) ^ 2
		end ./ length(ζhs)

		des["x"] = Array((bin[1:(end-1)] .+ bin[2:end]) / 2)
		des["y"] = Array(mean)
		des["Δy"] = Array(sqrt.(var))
	end,
	srcFiles[Re],
	create_group(desFile, "velocity-pdf:filtered/$(Re)"),
)

rect_sizes_moments(Re, order, width, height) = PostData(
	(ζhs, des) -> begin
		ns = 1:floor(Int, min(grid.nx/width, grid.ny/height))
		fh = lpFilter(Re)

		moments = map(ns) do n
			mean = sum(ζhs) do ζh
				Γ = rectCirculations(ζh .* fh, n*width, n*height, grid)
				sum(x->abs(x)^order, Γ) / length(Γ)
			end ./ length(ζhs)

			var = sum(ζhs) do ζh
				Γ = rectCirculations(ζh .* fh, n*width, n*height, grid)
				res = sum(x->abs(x)^order, Γ) / length(Γ)
                @. (res - mean) ^ 2
			end ./ length(ζhs)

			(mean, var)
		end

		des["x"] = Array(ns)
		des["y"] = Array(getindex.(moments, 1))
		des["Δy"] = Array(sqrt.(getindex.(moments, 2)))
	end,
	srcFiles[Re],
	create_group(desFile, "loop_sizes-rect_moments/$(Re)/$(order)/$(width)×$(height)"),
)

aspect_ratio_moments_area(Re, order) = PostData(
	(ζhs, des) -> begin
		fh = lpFilter(Re)

		ns = [15; 20; 30; 40; 45; 60; 90]

		moments = map(ns) do n
			mean = sum(ζhs) do ζh
				Γ = rectCirculations(ζh .* fh, n, 8100÷n, grid)
				sum(x->abs(x)^order, Γ) / length(Γ)
			end ./ length(ζhs)

			var = sum(ζhs) do ζh
				Γ = rectCirculations(ζh .* fh, n, 8100÷n, grid)
				res = sum(x->abs(x)^order, Γ) / length(Γ)
                @. (res - mean) ^ 2
			end ./ length(ζhs)

			(mean, var)
		end

		des["x"] = Array(ns .^ 2 ./ 8100)
		des["y"] = Array(getindex.(moments, 1))
		des["Δy"] = Array(sqrt.(getindex.(moments, 2)))
	end,
	srcFiles[Re],
	create_group(desFile, "aspect_ratio-moments/area8100/$(Re)/$(order)"),
)

aspect_ratio_moments_perimeter(Re, order) = PostData(
	(ζhs, des) -> begin
		fh = lpFilter(Re)

		ns = 10:10:90

		moments = map(ns) do n
			mean = sum(ζhs) do ζh
				Γ = rectCirculations(ζh .* fh, n, 180-n, grid)
				sum(x->abs(x)^order, Γ) / length(Γ)
			end ./ length(ζhs)

			var = sum(ζhs) do ζh
				Γ = rectCirculations(ζh .* fh, n, 108-n, grid)
				res = sum(x->abs(x)^order, Γ) / length(Γ)
                @. (res - mean) ^ 2
			end ./ length(ζhs)

			(mean, var)
		end

		des["x"] = Array(ns ./ (180 .- ns))
		des["y"] = Array(getindex.(moments, 1))
		des["Δy"] = Array(sqrt.(getindex.(moments, 2)))
	end,
	srcFiles[Re],
	create_group(desFile, "aspect_ratio-moments/perimeter180/$(Re)/$(order)"),
)

dataList = []

for Re in Reynolds
	push!(dataList,
		energy_spectrum(Re),
		energy_flux(Re),
		enstrophy_flux(Re),
		large_scale_velocity_pdf(Re),
	)

	for order in 1:10
		push!(dataList,
			rect_sizes_moments(Re, order, 10, 10),
			rect_sizes_moments(Re, order, 5, 20),
			rect_sizes_moments(Re, order, 4, 16),
			aspect_ratio_moments_area(Re, order),
			aspect_ratio_moments_perimeter(Re, order),
		)
	end

end

collect.(dataList)

close(desFile)
close.(values(srcFiles))








