function energy_spectrum(Re)
    datum = rawdata[Re]
    h5group = create_group(pfile, "wavenumber-energy_spectra/$(Re)")
    kr = radialk(datum.grid)
    Ehrs = map(datum.ζhs) do ζh
        Eh = energy(ζh, datum.grid)
        radialspectrum(Eh, datum.grid)
    end
    mean, std = measure(Ehrs)
    h5group["x"] = Array(kr)
    h5group["y"] = Array(mean)
    h5group["Δy"] = Array(std)
    return nothing
end

function energy_flux(Re)
    datum = rawdata[Re]
    h5group = create_group(pfile, "wavenumber-energy_fluxes/$(Re)")
    kr = radialk(datum.grid)
    Fhrs = map(datum.ζhs) do ζh
        Nh = energyInjectionRate(ζh, datum.grid)
        cumradialspectrum(Nh, datum.grid)
    end
    mean, std = measure(Fhrs)
    h5group["x"] = Array(kr)
    h5group["y"] = Array(mean)
    h5group["Δy"] = Array(std)
    return nothing
end

function enstrophy_flux(Re)
    datum = rawdata[Re]
    h5group = create_group(pfile, "wavenumber-enstrophy_fluxes/$(Re)")
    kr = radialk(datum.grid)
    kf = datum.grid.nx / 3sqrt(Re)
    Fhrs = map(datum.ζhs) do ζh
        Nh = enstrophyInjectionRate(ζh, datum.grid)
        cumradialspectrum(Nh, datum.grid) ./ kf^2
    end
    mean, std = measure(Fhrs)
    h5group["x"] = Array(kr)
    h5group["y"] = Array(mean)
    h5group["Δy"] = Array(std)
    return nothing
end

function velocity_pdf(Re, isfiltered)
    datum = rawdata[Re]
    if isfiltered
        h5group = create_group(pfile, "velocity-pdf:filtered/$(Re)")
        ζhs = datum.fζhs
    else
        h5group = create_group(pfile, "velocity-pdf:primitive/$(Re)")
        ζhs = datum.ζhs
    end
    grid = datum.grid

    umax = maximum(ζhs) do ζh
        uh = im * ζh .* grid.invKrsq .* grid.l
        u = grid.rfftplan \ uh
        maximum(u)
    end

    bin = range(-umax, umax, 100)
    bc = (bin[1:(end-1)] .+ bin[2:end]) / 2

    pdfs = map(ζhs) do ζh
        uh = im * ζh .* grid.invKrsq .* grid.l
        u = grid.rfftplan \ uh
        pdf(u, bin)
    end
    mean, std = measure(pdfs)
    h5group["x"] = Array(bc)
    h5group["y"] = Array(mean)
    h5group["Δy"] = Array(std)

    return nothing
end

function loopsizes_moments(Re, order, width, height)
    datum = rawdata[Re]
    grid = datum.grid
    ns = 1:floor(Int, min(grid.nx / width, grid.ny / height))
    h5group = create_group(pfile, "loop_sizes-square_moments/$(Re)/$(order)/$(width)×$(height)")
    loops = [rectsloop(grid, n * width, n * height, 1, 1) for n in ns]

    moments = map(loops) do hsh
        sp = map(datum.fζhs) do fζh
            Γ = getΓ(fζh, hsh, grid)
            moment(abs.(Γ), order)
        end
        mean, std = measure(sp)
        mean ± std
    end
    h5group["x"] = Array(ns)
    h5group["y"] = Array(Measurements.value.(moments))
    h5group["Δy"] = Array(Measurements.uncertainty.(moments))

    return nothing
end

function aspect_ratio_moments(Re, order, isarea)
    datum = rawdata[Re]
    grid = datum.grid
    if isarea
        h5group = create_group(pfile, "aspect_ratio-moments/area8100/$(Re)/$(order)")
        loopsizes = [15; 20; 30; 40; 45; 60; 90]
        rects = [rectsloop(grid, l, 8100 ÷ l, 1, 1) for l in loopsizes]
        aspect_ratios = loopsizes .^ 2 ./ 8100
    else
        h5group = create_group(pfile, "aspect_ratio-moments/perimeter180/$(Re)/$(order)")
        loopsizes = 10:10:90
        rects = [rectsloop(grid, l, 180 - l, 1, 1) for l in loopsizes]
        aspect_ratios = loopsizes ./ (180 .- loopsizes)
    end

    moments = map(rects) do hsh
        sp = map(datum.fζhs) do fζh
            Γ = getΓ(fζh, hsh, grid)
            moment(abs.(Γ), order)
        end
        mean, std = measure(sp)
        mean ± std
    end

    group["Re$(Re) order$(order)/x"] = Array(aspect_ratios)
    group["Re$(Re) order$(order)/y"] = Array(Measurements.value.(moments))
    group["Re$(Re) order$(order)/Δy"] = Array(Measurements.uncertainty.(moments))
end

@time for Re in keys(rawdata)
    energy_spectrum(Re)
    energy_flux(Re)
    enstrophy_flux(Re)
    velocity_pdf(Re, true)
    velocity_pdf(Re, false)
    for order in 1:10
        loopsizes_moments(Re, order, 10, 10)
        loopsizes_moments(Re, order, 5, 20)
        loopsizes_moments(Re, order, 4, 16)
        aspect_ratio_moments(Re, order, true)
        aspect_ratio_moments(Re, order, false)
    end
end